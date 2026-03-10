import argparse
import json
import os
import csv
import time

from vision_utils import (
    compute_path_a_metrics,
    load_bgr,
    save_debug_masks,
    draw_hud_overlay,
)
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
    # Path A (timed)
    t0 = time.time()
    path_a = compute_path_a_metrics(image_path)
    tA = time.time() - t0

    # Path B - CLIP + YOLO (timed together)
    t1 = time.time()
    clip = classify_clip(image_path, labels=None, topk=topk_clip)
    yolo = run_yolo(image_path, conf_thres=0.25, max_det=10)
    tB = time.time() - t1

    # Fusion
    report = fuse_report(path_a=path_a, clip=clip, yolo=yolo)

    # attach timing info under raw
    timing = {
        "path_a_seconds": round(float(tA), 6),
        "path_b_seconds": round(float(tB), 6),
    }
    raw = report.get("raw") or {}
    raw["timing"] = timing
    report["raw"] = raw

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

    # save masks + HUD
    bgr = load_bgr(image_path)
    if bgr is not None:
        base_out = os.path.join(out_dir, base)
        save_debug_masks(bgr, base_out)
        grade = str(report.get("quality_grade", "REVIEW"))
        draw_hud_overlay(
            bgr,
            path_a={**path_a, "_subject_summary": report.get("subject_summary", "not sure")},
            clip=clip,
            yolo=yolo,
            quality_grade=grade,
            out_path=base_out + "_HUD.jpg",
        )

    # tiny print (not spam)
    print(f"done: {image_path}")
    print(f" -> {out_path}")
    print(
        f"scene={report.get('final_scene')}  "
        f"grade={report.get('quality_grade')}  "
        f"green={path_a.get('green_coverage_percentage')}  "
        f"lap={path_a.get('laplacian_variance')}  "
        f"timeA={timing['path_a_seconds']:.4f}s  timeB={timing['path_b_seconds']:.4f}s"
    )

    return report


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

        rows = []
        for p in files:
            report = run_one(p, out_dir=args.out_dir, topk_clip=args.clip_topk, write_slim=not args.no_slim)
            base = os.path.basename(p)
            qa = report.get("quality_grade", "REVIEW")
            scene = report.get("final_scene", "unknown")
            pa = (report.get("raw") or {}).get("path_a_metrics", {}) or {}
            lap = pa.get("laplacian_variance", 0.0)
            edge = pa.get("edge_density_tuned", pa.get("edge_density", 0.0))
            yolo = (report.get("raw") or {}).get("yolo", {}) or {}
            counts = yolo.get("label_counts", {}) or {}
            if counts:
                # pick the most common label (HW-style simple loop)
                primary_label = None
                primary_count = -1
                for lab, cnt in counts.items():
                    if cnt > primary_count:
                        primary_label = lab
                        primary_count = cnt
            else:
                primary_label = "none"

            rows.append(
                {
                    "filename": base,
                    "quality_grade": qa,
                    "scene": scene,
                    "laplacian_variance": lap,
                    "edge_density_tuned": edge,
                    "primary_subject": primary_label,
                }
            )

        # write simple CSV report
        csv_path = os.path.join(args.out_dir, "master_culling_report.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "quality_grade", "scene", "laplacian_variance", "edge_density_tuned", "primary_subject"])
            for r in rows:
                w.writerow(
                    [
                        r["filename"],
                        r["quality_grade"],
                        r["scene"],
                        r["laplacian_variance"],
                        r["edge_density_tuned"],
                        r["primary_subject"],
                    ]
                )

        print(f"wrote CSV report: {csv_path}")
        return

    raise RuntimeError("pass --image or --input_dir")


if __name__ == "__main__":
    main()

