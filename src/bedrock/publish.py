"""Copy the pipeline outputs into app/snapshot/ so the dashboard runs on Streamlit Cloud
(which has no data/ folder: that is git-ignored).

    python -m bedrock.publish           # copy outputs -> app/snapshot/ (refuses if built from Rock data)
    python -m bedrock.publish --check   # open all 4 dashboard screens on the snapshot; exit 1 on any error

Safety: Rock Data Plus is licensed. If the outputs were built with files from data/rock/, publishing
is refused, so licensed numbers never reach a public repo by accident.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

from .config import PROCESSED, ROOT

OUT = PROCESSED / "outputs"
SNAP = ROOT / "app" / "snapshot"


def publish() -> None:
    head = OUT / "headline.json"
    if not head.exists():
        sys.exit("No outputs. Run: python -m bedrock.pipeline")
    meta = json.loads(head.read_text())
    if meta.get("uses_rock_data", False):
        sys.exit("Refused: these outputs were built with Rock Data Plus files (data/rock/). "
                 "Move them aside, rerun the pipeline on public data, then publish.")
    SNAP.mkdir(parents=True, exist_ok=True)
    for f in SNAP.iterdir():
        f.unlink()
    n = 0
    for f in OUT.iterdir():
        if f.suffix in (".parquet", ".json"):
            shutil.copy2(f, SNAP / f.name)
            n += 1
    print(f"Copied {n} files to {SNAP} (data as of {meta['data_as_of']['stocks']}). Commit app/snapshot/ to deploy.")


def check() -> None:
    from streamlit.testing.v1 import AppTest
    screens = ["Overview", "Market depth", "Sell-off alert", "Crash risk", "Speculation", "Stress test"]
    bad = 0
    for s in screens:
        at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
        at.button_group[0].set_value(s).run()           # top navigation (st.segmented_control)
        errs = [e.value for e in at.exception] + [e.value for e in at.error]
        print(f"  {s}: {'OK' if not errs else 'ERROR ' + str(errs)[:300]}")
        bad += bool(errs)
    if bad:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check() if a.check else publish()


if __name__ == "__main__":
    main()
