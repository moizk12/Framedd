from __future__ import annotations

from typing import Dict, Any, List


# Fusion Layer (hardcoded rules)
# this is the "anti hallucination leash" - simple if else based on evidence


def _has_yolo_label(yolo: Dict[str, Any], label: str) -> bool:
    cnts = yolo.get("label_counts", {}) or {}
    return cnts.get(label, 0) > 0


def fuse_report(path_a: Dict[str, Any], clip: Dict[str, Any], yolo: Dict[str, Any]) -> Dict[str, Any]:
    """
    combine path A + path B into final json report
    all rules are explicit so grading is easy
    """
    notes: List[str] = []

    green_pct = float(path_a.get("green_coverage_percentage", 0.0))
    lap_var = float(path_a.get("laplacian_variance", 0.0))
    bright = float(path_a.get("brightness_mean", 0.0))
    contrast = float(path_a.get("contrast_std", 0.0))
    # prefer tuned edge density (HW5), but keep fallback for backward compatibility
    edge_dens = float(path_a.get("edge_density_tuned", path_a.get("edge_density", 0.0)))
    if "edge_density_tuned" in path_a:
        notes.append("metric: using edge_density_tuned (HW5 tuned canny)")

    green_comp_count = int(path_a.get("green_component_count", 0) or 0)
    green_small_ratio = float(path_a.get("green_small_component_ratio", 0.0) or 0.0)

    scene = clip.get("primary_scene", "unknown")
    scene_conf = float(clip.get("confidence", 0.0))

    # simple buckets
    is_very_textured = lap_var >= 400.0
    is_low_texture = lap_var <= 120.0
    is_edge_heavy = edge_dens >= 0.12
    has_green = green_pct >= 5.0

    # subject presence (YOLO)
    has_person = _has_yolo_label(yolo, "person")
    has_car = _has_yolo_label(yolo, "car")
    has_building = _has_yolo_label(yolo, "building")
    subject_present = has_person or has_car or has_building

    # HW4/HW7 clutter / fragmentation notes (green mask spatial structure)
    if green_comp_count >= 25:
        notes.append("rule: green_component_count high -> fragmented organic regions / clutter signal")
    if green_small_ratio >= 0.6 and green_comp_count >= 5:
        notes.append("rule: green_small_component_ratio high -> many small organic regions (high fragmentation)")

    organic_integration = "some"
    if green_pct == 0.0:
        organic_integration = "none"
        notes.append("rule: green_coverage_percentage==0.0 -> organic_integration none")
    elif has_green:
        organic_integration = "present"

    # decide subject/summary
    subject_summary = "not sure"

    # YOLO overrides for "object exists"
    if has_person:
        notes.append("yolo: person detected")
    if has_car:
        notes.append("yolo: car detected")
    if has_building:
        notes.append("yolo: building detected")

    # track conflicts between semantics and pixels
    conflict_flag = False

    # fusion logic
    if scene == "urban architecture":
        if is_very_textured or is_edge_heavy:
            subject_summary = "highly textured manufactured structure"
            notes.append("rule: urban architecture + high texture/edges")
        else:
            subject_summary = "manufactured structure (low/mod texture)"
    elif scene in ["mountain landscape", "forest greenery", "beach ocean"]:
        if green_pct == 0.0 and scene in ["forest greenery"]:
            subject_summary = "semantic says forest but pixel green is none (prob fail case)"
            notes.append("rule: forest label but green==0 -> conflict")
            conflict_flag = True
        else:
            subject_summary = "natural landscape scene"
    elif scene == "indoor room":
        subject_summary = "indoor environment / room"
    elif scene == "street scene":
        subject_summary = "street / outdoor public space"
    elif scene == "portrait photo":
        subject_summary = "portrait style photo"
    elif scene == "night city":
        subject_summary = "nighttime city scene"
    elif scene == "abstract art":
        subject_summary = "abstract / artistic scene"

    texture_statement = "medium texture"
    if is_very_textured:
        texture_statement = "high texture (laplacian variance high)"
    elif is_low_texture:
        texture_statement = "low texture (laplacian variance low)"

    # final scene label (we mostly trust clip for this, but we log conflict)
    final_scene = scene
    if scene == "mountain landscape" and (has_building or has_car):
        notes.append("rule: clip says mountain but yolo sees man-made obj")
        conflict_flag = True

    # Hard quality grade
    quality_grade = "REVIEW"
    if lap_var < 150.0 or bright < 20.0:
        quality_grade = "REJECT"
        if lap_var < 150.0:
            notes.append("rule: quality_grade REJECT (very low laplacian_variance)")
        if bright < 20.0:
            notes.append("rule: quality_grade REJECT (very low brightness_mean)")
    else:
        high_sharp = lap_var >= 400.0
        good_contrast = contrast >= 25.0
        if high_sharp and good_contrast and subject_present:
            quality_grade = "PASS"
            notes.append("rule: quality_grade PASS (sharp, good contrast, clear subject)")

    # downgrade PASS to REVIEW if we saw a semantic vs pixel conflict
    if conflict_flag and quality_grade == "PASS":
        quality_grade = "REVIEW"
        notes.append("rule: quality_grade REVIEW (semantic vs pixel conflict)")

    out = {
        "final_scene": final_scene,
        "final_scene_confidence": round(scene_conf, 6),
        "subject_summary": subject_summary,
        "organic_integration": organic_integration,
        "texture_statement": texture_statement,
        "quality_grade": quality_grade,
        "raw": {
            "path_a_metrics": path_a,
            "clip": clip,
            "yolo": yolo,
        },
        "notes": notes,
    }

    return out

