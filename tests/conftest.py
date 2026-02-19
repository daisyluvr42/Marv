from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HEARTBEAT_ENABLED", "false")
os.environ.setdefault("EDGE_HEARTBEAT_CONFIG_PATH", f"/tmp/marv_test_heartbeat_{os.getpid()}.json")
