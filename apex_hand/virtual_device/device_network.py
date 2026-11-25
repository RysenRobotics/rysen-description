"""Manage the two TCP listeners and the SDK set-address/reset sequence."""

import ipaddress
import socket
import threading


def close_socket(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


class AddressBinding:
    def __init__(self, address, ports):
        self.address = address
        self.listeners = []
        self.clients = set()
        self.lock = threading.Lock()
        self.closed = False
        try:
            for port in ports:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.listeners.append(listener)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((address, port))
        except OSError:
            self.close()
            raise

    def listen(self):
        for listener in self.listeners:
            listener.listen(5)

    def serve(self, handlers):
        for listener, handler in zip(self.listeners, handlers):
            threading.Thread(target=self._serve, args=(listener, handler), daemon=True).start()

    def _serve(self, listener, handler):
        while True:
            try:
                client, _ = listener.accept()
            except OSError:
                return
            with self.lock:
                if self.closed:
                    client.close()
                    return
                self.clients.add(client)
            try:
                with client:
                    handler(client)
            except (OSError, ValueError) as error:
                if not self.closed and client in self.clients:
                    print(f"[{self.address}] 连接结束: {error}")
            finally:
                with self.lock:
                    self.clients.discard(client)

    def disconnect_clients(self):
        with self.lock:
            for client in self.clients:
                close_socket(client)
            self.clients.clear()

    def close(self):
        with self.lock:
            self.closed = True
            for listener in self.listeners:
                close_socket(listener)
        self.disconnect_clients()


class DeviceNetwork:
    def __init__(self, address, ports, reset_hand):
        self.ports = ports
        self.reset_hand = reset_hand
        self.active = AddressBinding(address, ports)
        self.pending = None
        self.reset_timer = None
        self.lock = threading.Lock()
        self.closed = False

    def start(self, control, feedback):
        self.handlers = (control, lambda client: feedback(client, self))
        self.active.listen()
        self.active.serve(self.handlers)

    def prepare_address(self, request):
        # SDK 1.6.0 supplies a fixed /24 mask and no gateway. Linux owns the interface addresses.
        if len(request) != 12 or request[4:] != bytes((255, 255, 255, 0, 0, 0, 0, 0)):
            return False
        address = ipaddress.IPv4Address(request[:4])
        if address.packed[0] in (0, 127) or address.packed[0] >= 224:
            return False
        host = str(address)
        with self.lock:
            if self.closed or self.reset_timer is not None:
                return False
            if self.pending and self.pending.address == host:
                return True
            replacement = None
            if self.active.address != host:
                try:
                    replacement = AddressBinding(host, self.ports)
                except OSError:
                    return False
            if self.pending:
                self.pending.close()
            self.pending = replacement
        return True

    def schedule_reset(self, delay_ms):
        with self.lock:
            if self.closed:
                return
            if self.reset_timer:
                self.reset_timer.cancel()
            self.reset_timer = threading.Timer(delay_ms / 1000.0, self._reset)
            self.reset_timer.daemon = True
            self.reset_timer.start()

    def _reset(self):
        with self.lock:
            if self.closed or self.reset_timer is not threading.current_thread():
                return
            self.reset_timer = None
            if self.pending is None:
                self.active.disconnect_clients()
                self.reset_hand()
                return

            replacement = self.pending
            self.pending = None
            try:
                replacement.listen()
            except OSError as error:
                replacement.close()
                print(f"改址后启动监听失败: {error}", flush=True)
                return
            self.active.close()
            self.reset_hand()
            self.active = replacement
            replacement.serve(self.handlers)
            print(f"[*] 仿真设备已切换地址: {replacement.address}", flush=True)

    def close(self):
        with self.lock:
            self.closed = True
            if self.reset_timer:
                self.reset_timer.cancel()
            if self.pending:
                self.pending.close()
            self.active.close()
