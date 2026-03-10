import argparse
import json
import os

from vision_utils import compute_path_a_metrics
from semantics_utils import classify_clip
from yolo_utils import run_yolo
from fusion_rules import fuse_report


# main runner
# 1) Path A numbers
# 2) Path B tags (clip) + yolo
# 3) Fusion report


def ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def make_slim_report(report: dict) -> dict:
    """
    Slide-friendly summary JSON (keep small: ~15-18 lines when pretty-printed).
    Keeps the core results without the big raw blobs.
    """
    raw = report.get("raw", {}) or {}
    path_a = raw.get("path_a_metrics", {}) or {}
    clip = raw.get("clip", {}) or {}
    yolo = raw.get("yolo", {}) or {}

    clip_top = clip.get("top_k", []) or []
    clip_top3 = {str(x.get("label")): float(x.get("confidence", 0.0)) for x in clip_top[:3]}

    return {
        "final_scene": report.get("final_scene", "unknown"),
        "final_scene_confidence": report.get("final_scene_confidence", 0.0),
        "subject_summary": report.get("subject_summary", "not sure"),
        "path_a": {
            "green_coverage_percentage": path_a.get("green_coverage_percentage", 0.0),
            "laplacian_variance": path_a.get("laplacian_variance", 0.0),
            "edge_density_tuned": path_a.get("edge_density_tuned", path_a.get("edge_density", 0.0)),
            "brightness_mean": path_a.get("brightness_mean", 0.0),
            "contrast_std": path_a.get("contrast_std", 0.0),
            "hist_entropy": path_a.get("hist_entropy", 0.0),
            "equalize_diff_energy": path_a.get("equalize_diff_energy", 0.0),
            "green_component_count": path_a.get("green_component_count", 0),
            "green_small_component_ratio": path_a.get("green_small_component_ratio", 0.0),
        },
        "semantics": {
            "clip_top3": clip_top3,
            "yolo_label_counts": yolo.get("label_counts", {}) or {},
        },
    }


def run_one(image_path: str, out_dir: str, topk_clip: int = 3, write_slim: bool = True):
    # Path A
    path_a = compute_path_a_metrics(image_path)

    # Path B - CLIP
    clip = classify_clip(image_path, labels=None, topk=topk_clip)

    # Path B - YOLO
    yolo = run_yolo(image_path, conf_thres=0.25, max_det=10)

    # Fusion
    report = fuse_report(path_a=path_a, clip=clip, yolo=yolo)

    # output file name
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if write_slim:
        slim = make_slim_report(report)
        slim_path = os.path.join(out_dir, f"{base}.slim.json")
        with open(slim_path, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2)

    # tiny print (not spam)
    print(f"done: {image_path}")
    print(f" -> {out_path}")
    print(f"scene={report.get('final_scene')}  green={path_a.get('green_coverage_percentage')}  lap={path_a.get('laplacian_variance')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None, help="path to 1 img")
    ap.add_argument("--input_dir", type=str, default=None, help="dir of imgs")
    ap.add_argument("--out_dir", type=str, default="sample_outputs", help="output json dir")
    ap.add_argument("--clip_topk", type=int, default=3)
    ap.add_argument("--no_slim", action="store_true", help="disable writing *.slim.json")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    if args.image:
        run_one(args.image, out_dir=args.out_dir, topk_clip=args.clip_topk, write_slim=not args.no_slim)
        return

    if args.input_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = []
        for fn in os.listdir(args.input_dir):
            p = os.path.join(args.input_dir, fn)
            if os.path.isfile(p) and os.path.splitext(fn.lower())[1] in exts:
                files.append(p)
        files.sort()

        if not files:
            raise RuntimeError("no image files found in input_dir")

        for p in files:
            run_one(p, out_dir=args.out_dir, topk_clip=args.clip_topk, write_slim=not args.no_slim)
        return

    raise RuntimeError("pass --image or --input_dir")


if __name__ == "__main__":
    main()

