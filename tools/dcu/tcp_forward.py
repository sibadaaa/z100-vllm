#!/usr/bin/env python3
"""Tiny TCP forwarder: host 127.0.0.1:<LPORT> -> container <CT_IP>:<RPORT>.

Used to expose a service that listens inside a bridge-networked docker container
on the host loopback, so the user can hit http://127.0.0.1:8000 directly.

Usage: tcp_forward.py <listen_host> <listen_port> <target_host> <target_port>
"""
import socket
import sys
import threading


def pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def handle(client: socket.socket, target: tuple[str, int]) -> None:
    try:
        upstream = socket.create_connection(target, timeout=10)
    except OSError:
        client.close()
        return
    threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()
    threading.Thread(target=pipe, args=(upstream, client), daemon=True).start()


def main() -> int:
    lhost, lport, thost, tport = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((lhost, lport))
    srv.listen(128)
    print(f"forwarding {lhost}:{lport} -> {thost}:{tport}", flush=True)
    while True:
        client, _ = srv.accept()
        threading.Thread(target=handle, args=(client, (thost, tport)), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
