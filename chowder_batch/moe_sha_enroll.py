"""Run one-time SHA256 enrollment from an uninstalled Chowder checkout.

Usage: python chowder_batch/moe_sha_enroll.py --model MODEL --out INVENTORY
This reads the complete checkpoint once, so call explicitly during qualification.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chowder.moe_checkpoint_inventory import main  # noqa: E402


if __name__ == "__main__":
    main()
