"""Drive one pdf2md live conversion via the REST API and wait for it.

Usage: python eval/live_convert.py <pdf-path> [timeout-minutes]
Prints job id and final status; exits non-zero on failure.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000/api/v1"
OUT_DIR = str(Path("output").resolve())


def post(path: str, data: bytes, headers: dict) -> dict:
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=120) as resp:
        return json.load(resp)


def main() -> int:
    pdf = Path(sys.argv[1]).resolve()
    timeout_min = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if not pdf.is_file():
        print(f"not found: {pdf}")
        return 2

    boundary = "pdf2mdevboundary42"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (
        'Content-Disposition: form-data; name="file"; '
        f'filename="{pdf.name}"\r\nContent-Type: application/pdf\r\n\r\n'
    ).encode()
    body += pdf.read_bytes() + b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += (f'Content-Disposition: form-data; name="output_dir"\r\n\r\n{OUT_DIR}\r\n').encode()
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="options"\r\n\r\n{}\r\n'
    body += f"--{boundary}--\r\n".encode()

    job = post(
        "/jobs",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    job_id = job["job_id"]
    print(f"job {job_id} started: {pdf.name}")

    deadline = time.time() + timeout_min * 60
    last_stage = ""
    while time.time() < deadline:
        time.sleep(20)
        detail = get(f"/jobs/{job_id}")
        stage = detail.get("stage", "")
        if stage != last_stage:
            print(
                f"  [{time.strftime('%H:%M:%S')}] stage={stage} pages={detail.get('pages_done')}/{detail.get('page_count')}"
            )
            last_stage = stage
        if not detail.get("live") and detail["status"] in (
            "completed",
            "failed",
            "cancelled",
            "paused",
        ):
            print(f"final status: {detail['status']}")
            if detail["status"] == "completed":
                print(f"output_dir: {detail['output_dir']}")
                return 0
            print(f"error: {detail.get('error')}")
            return 1
    print(f"TIMEOUT after {timeout_min} minutes — job {job_id} still running")
    return 3


if __name__ == "__main__":
    sys.exit(main())
