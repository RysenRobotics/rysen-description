import numpy as np
import json
import re
from pathlib import Path

import mujoco
from scipy.spatial.transform import Rotation

# 锚在 examples/ 而不是靠 .parent 层数, 这个文件挪进 lib/ 后不必再数目录
EXAMPLES_DIR = Path(__file__).resolve().parent.parent
# 触觉点阵的排布表(与实物传感器一致), 见 urdf/README.md 的 Tactile points 一节
TACTILE_REMAP = json.loads(
    (EXAMPLES_DIR.parent.parent / "urdf" / "tactile" / "remap.json")
    .read_text(encoding="utf-8"))

# 单个触点的力超过这个值后, 箭头长度不再增加(牛顿)
TACTILE_FORCE_MAX = 2.0


def _remap_key(body_name):
    """把 MJCF 的 pad 名映射到 remap.json 里的键。

    ``right_index_tactile2`` -> ``("finger", "3")``; ``right_thumb_tactile0``
    -> ``("thumb", "2")``; ``right_palm_tactile`` -> ``("palm", "right")``。

    四指的 pad 0/1/2 依次是 finger.1/2/3, 拇指的 pad 0/1/2 依次是 thumb.2/3/4
    (拇指没有近节指腹), 两者的编号并不共用一套偏移。
    """
    match = re.fullmatch(r"(left|right)_palm_tactile", body_name)
    if match:
        return "palm", match.group(1)

    match = re.fullmatch(
        r"(left|right)_(thumb|index|middle|ring|pinky)_tactile(\d+)", body_name)
    if match is None:
        return None
    _, finger, pad = match.groups()
    if finger == "thumb":
        return "thumb", str(int(pad) + 2)
    return "finger", str(int(pad) + 1)



class RobotTactiles:
    def __init__(self, model: mujoco.MjModel):
        """收集模型里的触觉球, 按所属 pad body 分组。"""
        self.model = model

        geoms_by_body = {}
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name and "_point_" in name and not name.endswith("_visual"):
                geoms_by_body.setdefault(int(model.geom_bodyid[geom_id]), []).append(
                    (geom_id, name))

        # self.tactile_geoms: 按 pad body 名索引,
        #   count    -- 该 body 上的触点数
        #   names    -- geom 名
        #   ids      -- 从 _point_NNN 提取的点号
        #   geom_ids -- MuJoCo 的 geom id
        #   areas    -- 每个触点的等效受力面积 [m^2]
        # 四项一一对应、顺序一致。
        self.tactile_geoms = {}
        for body_id in sorted(geoms_by_body):
            geoms = geoms_by_body[body_id]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            self.tactile_geoms[body_name] = {
                "count": len(geoms),
                "names": [name for _, name in geoms],
                "ids": [int(re.search(r"_point_(\d+)$", name).group(1))
                        for _, name in geoms],
                "geom_ids": [geom_id for geom_id, _ in geoms],
            }

        # 等效受力面积: 触点(球面)到最近邻触点的距离平方, 即 Voronoi 微元的方形
        # 近似(边长=邻居间距)。同一 body 上的触点共享 link 局部坐标系。
        for info in self.tactile_geoms.values():
            ids = np.asarray(info["geom_ids"])
            normals = Rotation.from_quat(
                model.geom_quat[ids][:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
            points = model.geom_pos[ids] + model.geom_size[ids, 0][:, None] * normals
            dist = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            np.fill_diagonal(dist, np.inf)
            info["areas"] = dist.min(axis=1) ** 2

        total = sum(info["count"] for info in self.tactile_geoms.values())
        print(f"tactile bodies: {len(self.tactile_geoms)}, total tactile geoms: {total}")
        for body_name, info in self.tactile_geoms.items():
            print(f"  {body_name}: {info['count']} geoms, "
                  f"point ids {min(info['ids'])}-{max(info['ids'])}")

    def get_tactile_force(self, data: mujoco.MjData):
        """
        获取每个触点的法向分力(模拟传感器输出, 每个触点一个数值)

        self.tactile_forces 数据结构 (dict, 按body name索引, 与tactile_geoms的key一致):
        {
            "left_palm_link": np.array shape (count,),   # 每个触点的法向分力[N]
            ...                                             # 与tactile_geoms[body]["geom_ids"]顺序一一对应
        }
        无接触的触点为0; 有接触则为作用在触觉球上的接触力在触点法向(z轴)方向的分力大小
        (同一触点的多个接触累加)

        同步输出 self.tactile_maps_kpa: 按 remap 排布成二维压强图(kPa),
        无元件的格子为 NaN
        """
        self.tactile_forces = {
            body_name: np.zeros(info["count"])
            for body_name, info in self.tactile_geoms.items()
        }
        # geom_id -> (body name, 该body内触点索引)
        geom_to_index = {
            geom_id: (body_name, k)
            for body_name, info in self.tactile_geoms.items()
            for k, geom_id in enumerate(info["geom_ids"])
        }

        for i in range(data.ncon):
            contact = data.contact[i]
            result = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, result)
            for geom_id, sign in ((contact.geom1, -1.0), (contact.geom2, 1.0)):
                entry = geom_to_index.get(geom_id)
                if entry is None:
                    continue
                force = sign * (contact.frame.reshape(3, 3).T @ result[0:3])
                # 触点法向: geom世界姿态的z轴(球心指向触点表面方向)
                normal = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
                body_name, k = entry
                self.tactile_forces[body_name][k] += abs(float(np.dot(force, normal)))

        # 缓存各触点的球面位置与法向, 供可视化使用
        self.tactile_points = {}
        self.tactile_normals = {}
        for body_name, info in self.tactile_geoms.items():
            ids = np.asarray(info["geom_ids"])
            xmat = data.geom_xmat[ids].reshape(-1, 3, 3)
            radius = self.model.geom_size[ids, 0]
            self.tactile_points[body_name] = data.geom_xpos[ids] + radius[:, None] * xmat[:, :, 2]
            self.tactile_normals[body_name] = xmat[:, :, 2]

        # 按 remap 排布成二维压强图(单位kPa), 与实物传感器的输出布局一致
        self.tactile_maps_kpa = {}
        for body_name, info in self.tactile_geoms.items():
            table = self._remap_table(body_name)
            if table is None:
                continue
            # 压强[Pa] = 法向力[N] / 等效面积[m^2]
            pressures = self.tactile_forces[body_name] / info["areas"]
            image = np.full(table.shape, np.nan)
            valid = table >= 0
            image[valid] = pressures[table[valid]] / 1000.0
            self.tactile_maps_kpa[body_name] = image

    @staticmethod
    def _remap_table(body_name):
        """Return the remap grid for a pad, or None if it has no entry.

        remap.json is nested by part and then by index:
        ``{"palm": {"right": ...}, "thumb": {"2": ...}, "finger": {"3": ...}}``.
        """
        key = _remap_key(body_name)
        if key is None:
            return None
        part, index = key
        group = TACTILE_REMAP.get(part)
        if group is None:
            return None
        table = group.get(index)
        return None if table is None else np.asarray(table)

    def draw_tactile_force(self, scene):
        """
        把每个触点的法向力画成红色箭头, 追加到 scene(可以是 viewer.user_scn,
        也可以是 Renderer.scene)。需先调用 get_tactile_force。

        箭头沿触点法向, 长度正比于力大小, 力为 0 的触点不画; 长度在
        TACTILE_FORCE_MAX 处饱和, 再大的力也不再变长。

        追加而非覆盖: 调用方决定 scene.ngeom 从哪开始, 这样离屏渲染时场景里
        已经放好的灯光和地面不会被清掉。
        """
        scale = 0.2  # 力到长度的缩放(1N -> 2mm)
        width = 0.001  # 箭头粗细(米)

        num = scene.ngeom
        for body_name, forces in self.tactile_forces.items():
            for k in np.nonzero(forces)[0]:
                start = self.tactile_points[body_name][k]
                # Clamped: the grasp routinely loads a single point past 2 N, and
                # letting the arrow keep growing would swamp the scene and hide
                # the distribution that matters.
                magnitude = min(forces[k], TACTILE_FORCE_MAX)
                end = start - magnitude * scale * self.tactile_normals[body_name][k]
                geom = scene.geoms[num]
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3), np.zeros(3), np.zeros(9),
                    np.array([1.0, 0.0, 0.0, 1.0]))
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)
                num += 1
        scene.ngeom = num

    def visualize_tactile_force(self, viewer):
        """把箭头画进交互式窗口, 覆盖上一帧的箭头。"""
        viewer.user_scn.ngeom = 0
        self.draw_tactile_force(viewer.user_scn)
