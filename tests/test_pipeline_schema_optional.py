from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "moiz_kashif_csc481_final" / "dataset_v2"


RUN = os.environ.get("RUN_SEMANTICS_TESTS", "").strip() == "1"


@pytest.mark.skipif(not RUN, reason="set RUN_SEMANTICS_TESTS=1 to run CLIP+YOLO integration tests")
def test_pipeline_output_schema_contains_semantics_and_notes():
    from moiz_kashif_csc481_final.pipeline import run_one

    # pick a stable sample image
    img = DATASET_DIR / "nature" / "v2_nature_001.jpg"
    if not img.exists():
        pytest.skip(f"missing image: {img}")

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        run_one(str(img), out_dir=str(out_dir), topk_clip=3)
        out_json = out_dir / f"{img.stem}.json"
        assert out_json.exists()
        data = json.loads(out_json.read_text(encoding="utf-8"))

    assert "final_scene" in data
    assert "raw" in data and isinstance(data["raw"], dict)
    assert "clip" in data["raw"]
    assert "yolo" in data["raw"]
    assert "notes" in data and isinstance(data["notes"], list)

