#!/usr/bin/env python3
"""Parallel range downloader for one HF file, with per-chunk length checks.

S8b / S9 finding: the HF `resolve/main` endpoints can advertise a bogus
(shorter) Content-Length, which makes `curl -C -` shrink the partial file.
Range requests are still honoured, so download fixed byte ranges in parallel
and verify every chunk length, then verify the whole-file sha256.

usage: par_dl.py <url> <out> <total_size> <sha256> [threads] [chunk_mb]
"""

import hashlib
import os
import sys
import threading
import time

import requests

URL = sys.argv[1]
OUT = sys.argv[2]
TOTAL = int(sys.argv[3])
SHA = sys.argv[4]
THREADS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
CHUNK = (int(sys.argv[6]) if len(sys.argv) > 6 else 8) * 1024 * 1024

lock = threading.Lock()
done = 0
failed = []


def fetch(offset: int, length: int, attempt: int = 0) -> None:
    global done
    end = min(offset + length, TOTAL) - 1
    headers = {"Range": f"bytes={offset}-{end}"}
    try:
        with requests.get(URL, headers=headers, stream=True, timeout=(20, 120)) as r:
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            buf = bytearray()
            for piece in r.iter_content(1024 * 256):
                buf += piece
            if len(buf) != end - offset + 1:
                raise RuntimeError(f"short chunk: got {len(buf)} want {end - offset + 1}")
        with open(OUT, "r+b") as fh:
            fh.seek(offset)
            fh.write(buf)
        with lock:
            done += end - offset + 1
            print(f"  {done}/{TOTAL} ({100.0 * done / TOTAL:.1f}%)", flush=True)
    except Exception as exc:
        if attempt >= 5:
            with lock:
                failed.append((offset, length, repr(exc)))
            print(f"  FAIL chunk {offset}+{length}: {exc!r}", flush=True)
            return
        time.sleep(2 * (attempt + 1))
        fetch(offset, length, attempt + 1)


def main() -> int:
    if os.path.exists(OUT) and os.path.getsize(OUT) == TOTAL:
        print("size already correct, verifying sha256 only")
    else:
        with open(OUT, "wb") as fh:
            fh.truncate(TOTAL)
    chunks = [(o, min(CHUNK, TOTAL - o)) for o in range(0, TOTAL, CHUNK)]
    print(f"downloading {TOTAL} B in {len(chunks)} chunks, {THREADS} threads")
    queue = list(chunks)
    qlock = threading.Lock()

    def worker():
        while True:
            with qlock:
                if not queue:
                    return
                offset, length = queue.pop(0)
            fetch(offset, length)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREADS)]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        print(f"{len(failed)} chunks failed", flush=True)
        return 1
    h = hashlib.sha256()
    with open(OUT, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    got = h.hexdigest()
    ok = got == SHA
    print(f"sha256 {got} expected {SHA} match={ok} "
          f"size={os.path.getsize(OUT)} elapsed={time.time() - started:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
