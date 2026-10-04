"""Wait until GPU 0 has at least MIN_FREE_MIB free, then exit 0.

Parse-failure-proof: nvidia-smi is read with nounits and stripped to digits;
any parse failure keeps waiting. This loop only waits - it never kills or
reconfigures the resident inference servers.
"""
import subprocess
import sys
import time
from datetime import datetime, timezone

MIN_FREE_MIB = 8192


def free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=30)
        parts = [p for p in out.stdout.replace("\r", "").strip().split(",")]
        digits = ["".join(c for c in p if c.isdigit()) for p in parts]
        if len(digits) != 2 or not all(digits):
            return None
        return int(digits[1]) - int(digits[0])
    except Exception:  # noqa: BLE001 - any failure keeps waiting
        return None


def main() -> int:
    while True:
        free = free_mib()
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        if free is None:
            print(f"{stamp} parse failure; keep waiting", flush=True)
        elif free >= MIN_FREE_MIB:
            print(f"{stamp} {free} MiB free >= {MIN_FREE_MIB}; go", flush=True)
            return 0
        else:
            print(f"{stamp} {free} MiB free < {MIN_FREE_MIB}; waiting", flush=True)
        time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())
