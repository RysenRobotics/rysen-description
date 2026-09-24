#!/usr/bin/env python3
"""Apex hand URDF 查看器: 一条命令在 RViz 里看模型, 无需 colcon 构建.

    python3 rviz/rviz_viewer.py                          # 左手, 静态 0 位
    python3 rviz/rviz_viewer.py --side right --tactile   # 右手带触点
    python3 rviz/rviz_viewer.py --slider                 # 加关节滑条窗口

依赖: ROS 2(robot_state_publisher, rviz2), rclpy; --slider 另需 tkinter.
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]  # description/apex_hand
URDF_DIR = PKG_DIR / "urdf"
RVIZ_TEMPLATE = Path(__file__).resolve().parent / "view.rviz"

SDK_JOINTS = [  # 与 URDF 同名(加侧别前缀), 共 21 个
    *[f"thumb_j{i}" for i in range(5)],
    *[f"{f}_j{i}" for f in ("index", "middle", "ring", "pinky") for i in range(4)],
]
COUPLED = {"thumb_j4": "thumb_j3", "index_j3": "index_j2", "middle_j3": "middle_j2",
           "ring_j3": "ring_j2", "pinky_j3": "pinky_j2"}  # 影子关节跟随主关节
INDEPENDENT = [j for j in SDK_JOINTS if j not in COUPLED]  # 滑条控制的 16 个


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=("left", "right"), default="left", help="左右手 (默认 left)")
    p.add_argument("--tactile", action="store_true", help="加载带触点的 *_tactile.urdf")
    p.add_argument("--slider", action="store_true", help="启动关节滑条窗口 (默认仅静态 0 位)")
    p.add_argument("--_role", choices=("gui", "static"), default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def load_urdf(side: str, tactile: bool) -> str:
    name = f"apex_hand_{side}{'_tactile' if tactile else ''}.urdf"
    path = URDF_DIR / f"apex_hand_{side}" / name
    if not path.exists():
        sys.exit(f"URDF 不存在: {path}")
    return path.read_text(encoding="ascii")


def rewrite_mesh_paths(text: str, side: str) -> str:
    """meshes/x.STL -> file:// 绝对路径, 仅作用于传给 RSP 的副本, 仓库 URDF 零改动."""
    mesh_dir = (URDF_DIR / f"apex_hand_{side}" / "meshes").resolve()
    return re.sub(r'filename="meshes/([^"]+)"', rf'filename="file://{mesh_dir}/\1"', text)


def joint_limits(side: str) -> dict[str, tuple[float, float]]:
    """从 base URDF 读各可动关节的上下限."""
    out = {}
    for j in ET.fromstring(load_urdf(side, tactile=False)).findall("joint"):
        if j.get("type") != "fixed":
            lim = j.find("limit")
            out[j.get("name").removeprefix(f"{side}_")] = (
                float(lim.get("lower")), float(lim.get("upper")))
    return out


# ---------------------------------------------------------------- 关节发布子进程

def run_publisher(args: argparse.Namespace) -> None:
    """发布 /joint_command: _role=gui 带滑条窗口, static 为静态 0 位.

    position 必须是 float(int 会被 setter 断言拒绝), stamp 必须是当前时间
    (零戳消息会被 RSP 的限流逻辑永久丢弃), 否则 RSP 不发可动关节的 TF.
    """
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    side, gui = args.side, args._role == "gui"
    values = {j: 0.0 for j in SDK_JOINTS}
    rclpy.init()
    node = Node("apex_joints")
    pub = node.create_publisher(JointState, "/joint_command", 10)

    def publish() -> None:
        if not node.context.ok():  # 信号到达后 rclpy 已关闭
            if gui:
                root.quit()
            return
        for shadow, main in COUPLED.items():
            values[shadow] = values[main]
        msg = JointState()
        msg.name = [f"{side}_{j}" for j in SDK_JOINTS]
        msg.position = [values[j] for j in SDK_JOINTS]
        msg.header.stamp = node.get_clock().now().to_msg()
        try:
            pub.publish(msg)
        except Exception:  # 与信号到达的退出竞态: context 恰在检查后被关闭
            pass  # 下一次循环/ tick 会看到 context 不 ok 而干净退出

    if not gui:  # 静态 0 位
        try:
            while node.context.ok():
                publish()
                time.sleep(0.05)  # 20 Hz
        finally:
            if node.context.ok():  # 信号到达时 rclpy 已关闭, 不能重复 shutdown
                rclpy.shutdown()
        return

    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title(f"Apex hand ({side}) — 关节滑条")
    scale = 100.0  # 滑条精度 0.01 rad
    scales = {}

    def on_slide(bare: str, v: str) -> None:
        values[bare] = float(v) / scale

    limits = joint_limits(side)
    for bare in INDEPENDENT:
        lo, hi = limits[bare]
        row = ttk.Frame(root)
        row.pack(fill="x", padx=6, pady=1)
        ttk.Label(row, text=bare, width=12, anchor="w").pack(side="left")
        scales[bare] = tk.Scale(row, from_=lo * scale, to=hi * scale, resolution=1,
                                orient="horizontal", showvalue=False,
                                command=lambda v, b=bare: on_slide(b, v))
        scales[bare].pack(side="left", fill="x", expand=True)

    def reset():  # set() 会触发 on_slide, values 随之归零
        for s in scales.values():
            s.set(0.0)

    ttk.Button(root, text="归零", command=reset).pack(pady=3)

    def tick():  # 20 Hz 持续发布, 后启动的 RViz 也能收到
        publish()
        root.after(50, tick)

    root.protocol("WM_DELETE_WINDOW", root.quit)
    tick()
    root.mainloop()
    if node.context.ok():
        rclpy.shutdown()


# ---------------------------------------------------------------- 主编排

def ros_cmd(cmd: list) -> list:
    """经 bash source ROS 环境; shlex.quote 防参数(含 JSON 空格)被 shell 拆散."""
    source = "for f in /opt/ros/*/setup.bash; do . $f; done 2>/dev/null"
    return ["bash", "-c", f"{source}; exec " + " ".join(shlex.quote(c) for c in cmd)]


def kill_tree(p, sig: int) -> None:
    """子进程自成进程组(start_new_session), 整组收, 防止孙进程变孤儿."""
    try:
        os.killpg(p.pid, sig)
    except ProcessLookupError:
        pass


def main() -> None:
    args = parse_args()
    if args._role:  # 子进程: 关节发布器
        run_publisher(args)
        return

    # RSP 只认 robot_description 参数(URDF 文件路径作位置参数不被接受, tactile 版
    # 554 KiB 也超命令行单参数上限); JSON 字符串恰好是合法的 YAML 双引号标量
    params = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    params.write("/**:\n  ros__parameters:\n    robot_description: ")
    params.write(json.dumps(rewrite_mesh_paths(load_urdf(args.side, args.tactile), args.side)))
    params.write("\n")
    params.close()

    rviz_tmp = tempfile.NamedTemporaryFile("w", suffix=".rviz", delete=False, encoding="utf-8")
    rviz_tmp.write(RVIZ_TEMPLATE.read_text(encoding="utf-8")
                   .replace("left_palm_link", f"{args.side}_palm_link"))
    rviz_tmp.close()

    # 三个组件: 关节发布器(滑条或静态) / RSP / RViz; 子进程输出静音(RSP 段日志、
    # rviz 的 libGL 报错等对看模型毫无价值)
    self_py = Path(__file__).resolve()
    role = "gui" if args.slider else "static"
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    procs = [
        subprocess.Popen(ros_cmd([sys.executable, str(self_py), "--_role", role,
                                  "--side", args.side]),
                         start_new_session=True, **quiet),
        subprocess.Popen(ros_cmd(["ros2", "run", "robot_state_publisher", "robot_state_publisher",
                                  "--ros-args", "--params-file", params.name,
                                  "-r", "/joint_states:=/joint_command"]),
                         start_new_session=True, **quiet),
        subprocess.Popen(ros_cmd(["ros2", "run", "rviz2", "rviz2", "-d", rviz_tmp.name]),
                         start_new_session=True, **quiet),
    ]

    print(f"[viewer] {args.side}{' +tactile' if args.tactile else ''} | "
          f"{'滑条窗口控制关节' if args.slider else '静态 0 位'} | Ctrl+C 退出")
    signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))  # 清理统一放在 finally

    try:
        while all(p.poll() is None for p in procs):  # 任一组件退出即整体收摊
            time.sleep(0.5)
        # 兜底提示: 用户关 RViz(rc=0)/关滑条窗口是正常退出, 除此之外的死亡(如 ROS 未装/
        # 无 DISPLAY)若不提示, 窗口不出现且无任何报错, 无从排查
        if (procs[1].poll() is not None or procs[2].returncode
                or (role == "static" and procs[0].poll() is not None)):
            print("[viewer] 组件异常退出, 终止本次运行", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # 清理期间忽略后续信号,
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # 防止二次 Ctrl+C 打断清理留下孤儿
        for p in procs:  # 先礼后兵, 整组收
            kill_tree(p, signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                kill_tree(p, signal.SIGKILL)
        for f in (params.name, rviz_tmp.name):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
