## Hybrid pipeline (CV + CLIP + YOLO)  class folder

This folder is separate from the original FRAMED repo code. It just runs a small 3-step pipeline and prints/saves a JSON report.

### What it does

- Path A (classical CV): gets deterministic metrics from OpenCV
  - green coverage % (HSV mask)
  - laplacian variance (texture proxy)
  - canny edge density (structure proxy)
- Path B (semantic): two small pretrained models
  - CLIP zero-shot scene tag (hardcoded label list)
  - YOLOv8n detections (objects + boxes)
- Fusion: hardcoded rules (if/else) combine it into final report + notes

### Install

From repo root:

```bash
pip install -r moiz_kashif_csc481_final/requirements.txt
```

### Run

Run 1 image:

```bash
python moiz_kashif_csc481_final/pipeline.py --image "path/to/img.jpg" --out_dir moiz_kashif_csc481_final/sample_outputs
```

Run a folder:

```bash
python moiz_kashif_csc481_final/pipeline.py --input_dir moiz_kashif_csc481_final/dataset/success_cases --out_dir moiz_kashif_csc481_final/sample_outputs
```

### Extra outputs for presentation

For each image, the pipeline writes:

- `*.json` – full report (Path A metrics, CLIP, YOLO, quality_grade, timing)
- `*.slim.json` – compact slide-friendly summary
- `*_green_mask.png` – HSV green mask (vegetation pixels)
- `*_edges.png` – Canny edge map (structure)
- `*_hud.jpg` – visual HUD overlay (YOLO boxes, CLIP scene, LapVar/edges, PASS/REVIEW/REJECT)

When running on a folder (`--input_dir`), a simple CSV is also written:

- `master_culling_report.csv` in the `out_dir`, with columns:
  - `filename`, `quality_grade`, `scene`, `laplacian_variance`, `edge_density_tuned`, `primary_subject`

### Auto-tests (HW-style rubric)

Run the deterministic CV-only tests (fast, no model downloads):

```bash
pytest -q moiz_kashif_csc481_final/tests
```

Run the optional CLIP+YOLO integration test (may download weights/models on first run):

```bash
RUN_SEMANTICS_TESTS=1 pytest -q moiz_kashif_csc481_final/tests/test_pipeline_schema_optional.py
```

### Notes

- YOLO uses `yolov8n.pt` and will auto-download it on first run.
- CLIP labels are fixed inside `semantics_utils.py` (so its not open ended).

