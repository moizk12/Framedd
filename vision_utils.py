import cv2
import numpy as np


# Path A  (classical CV stuff)
# keep it deterministic (same img -> same nums)

def _to_gray_u8(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to uint8 grayscale."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def load_bgr(image_path: str):
    """read img w opencv (BGR). return None if path bad"""
    img = cv2.imread(image_path)
    return img


def green_coverage_hsv(bgr: np.ndarray):
    """
    rough green/veg coverage using HSV mask
    returns: pct (0-100), plus mask for debug if needed
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # green-ish range (can tweak a bit if needed)
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([85, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)  # 0/255

    # coverage
    total = mask.size
    green_px = int(np.count_nonzero(mask))
    pct = 100.0 * (green_px / float(total))

    return pct, mask


def laplacian_variance(bgr: np.ndarray):
    """texture/sharpness proxy (bigger var -> more texture / sharper edges)"""
    gray = _to_gray_u8(bgr)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def histogram_stats(gray_u8: np.ndarray):
    """
    HW2-ish histogram summary:
    - entropy: distribution spread
    - peakiness: max bin probability
    """
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).reshape(-1).astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 0.0, 0.0
    p = hist / total
    eps = 1e-12
    entropy = float(-(p * np.log2(p + eps)).sum())
    peakiness = float(p.max())
    return entropy, peakiness


def equalize_diff_energy(gray_u8: np.ndarray):
    """
    HW2-ish: equalize histogram and measure how much it changes the image.
    Returns mean absolute diff in [0,255].
    """
    eq = cv2.equalizeHist(gray_u8)
    diff = cv2.absdiff(gray_u8, eq)
    return float(np.mean(diff))


def edge_density_canny(bgr: np.ndarray, t1=100, t2=200):
    """edge density via canny (0-1)"""
    gray = _to_gray_u8(bgr)
    edges = cv2.Canny(gray, threshold1=t1, threshold2=t2)
    total = edges.size
    edge_px = int(np.count_nonzero(edges))
    dens = edge_px / float(total)
    return float(dens), edges


def edge_density_canny_gray(gray_u8: np.ndarray, t1=100, t2=200):
    """edge density via canny given uint8 gray (0-1)"""
    edges = cv2.Canny(gray_u8, threshold1=t1, threshold2=t2)
    total = edges.size
    edge_px = int(np.count_nonzero(edges))
    dens = edge_px / float(total)
    return float(dens), edges


def basic_stats(bgr: np.ndarray):
    """extra simple stats (brightness/contrast) just in case"""
    gray = _to_gray_u8(bgr)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    return mean_val, std_val


def blur_sensitivity_edge_drop(gray_u8: np.ndarray, t1=100, t2=200):
    """
    HW3-ish: apply simple blurs and see how much edge density drops.
    Higher drop => more fine texture/noise / blur-sensitive detail.
    """
    dens0, _ = edge_density_canny_gray(gray_u8, t1=t1, t2=t2)
    mean_blur = cv2.blur(gray_u8, (3, 3))
    med_blur = cv2.medianBlur(gray_u8, 3)
    dens_mean, _ = edge_density_canny_gray(mean_blur, t1=t1, t2=t2)
    dens_med, _ = edge_density_canny_gray(med_blur, t1=t1, t2=t2)
    return float(dens0 - dens_mean), float(dens0 - dens_med)


def tuned_canny_edge_density(gray_u8: np.ndarray, sigmas=None, t1=100, t2=200, target_band=(0.04, 0.12)):
    """
    HW5-ish: sweep sigma values for Gaussian blur before Canny,
    choose sigma whose edge density is in target band (or closest to band midpoint).
    """
    if sigmas is None:
        sigmas = [1.0, 1.5, 2.0, 2.5, 3.0]
    lo, hi = float(target_band[0]), float(target_band[1])
    mid = (lo + hi) / 2.0

    dens_by_sigma = {}
    best_sigma = float(sigmas[0])
    best_score = float("inf")

    for s in sigmas:
        s = float(s)
        blurred = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=s, sigmaY=s)
        dens, _ = edge_density_canny_gray(blurred, t1=t1, t2=t2)
        dens_by_sigma[str(s)] = float(dens)
        # score: 0 if in band, else distance to band edge; tie-break to closeness to midpoint
        if dens < lo:
            dist = lo - dens
        elif dens > hi:
            dist = dens - hi
        else:
            dist = 0.0
        score = dist * 10.0 + abs(dens - mid)  # keep it simple/deterministic
        if score < best_score:
            best_score = score
            best_sigma = s

    chosen_dens = float(dens_by_sigma.get(str(best_sigma), list(dens_by_sigma.values())[0]))
    return float(best_sigma), float(chosen_dens), dens_by_sigma


def morph_cleanup(mask_u8: np.ndarray, ksize: int = 5):
    """HW4/HW7-ish: closing then opening on a binary mask (0/255)."""
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def connected_component_stats(mask_u8: np.ndarray):
    """
    Connected components stats for a binary mask (0/255).
    Returns: (count, top5_areas, small_component_ratio)
    """
    bin01 = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    comp_count = max(0, int(num) - 1)  # exclude background
    if comp_count == 0:
        return 0, [], 0.0

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)  # exclude background row
    areas_sorted = sorted([int(a) for a in areas.tolist()], reverse=True)
    top5 = areas_sorted[:5]

    total_px = int(mask_u8.size)
    small_thresh = max(30, int(0.001 * total_px))
    small_count = int(np.sum(areas < small_thresh))
    small_ratio = float(small_count / float(comp_count)) if comp_count > 0 else 0.0
    return comp_count, top5, small_ratio


def compute_path_a_metrics_from_bgr(bgr: np.ndarray):
    """Compute Path A metrics from an already-loaded BGR image."""
    gray = _to_gray_u8(bgr)

    green_pct, green_mask = green_coverage_hsv(bgr)
    green_clean = morph_cleanup(green_mask, ksize=5)
    green_comp_count, green_top5, green_small_ratio = connected_component_stats(green_clean)

    lap_var = laplacian_variance(bgr)
    edge_dens, _ = edge_density_canny_gray(gray)
    bright, contrast = float(np.mean(gray)), float(np.std(gray))

    hist_ent, hist_peak = histogram_stats(gray)
    eq_diff = equalize_diff_energy(gray)

    drop_mean, drop_med = blur_sensitivity_edge_drop(gray)

    canny_sigma, edge_dens_tuned, edge_by_sigma = tuned_canny_edge_density(gray)

    out = {
        # existing
        "green_coverage_percentage": round(float(green_pct), 4),
        "laplacian_variance": round(float(lap_var), 4),
        "edge_density": round(float(edge_dens), 6),
        "brightness_mean": round(float(bright), 4),
        "contrast_std": round(float(contrast), 4),
        # HW2
        "hist_entropy": round(float(hist_ent), 6),
        "hist_peakiness": round(float(hist_peak), 8),
        "equalize_diff_energy": round(float(eq_diff), 6),
        # HW3
        "edge_drop_mean_blur": round(float(drop_mean), 6),
        "edge_drop_median_blur": round(float(drop_med), 6),
        # HW5
        "canny_sigma_chosen": float(canny_sigma),
        "edge_density_tuned": round(float(edge_dens_tuned), 6),
        "edge_density_by_sigma": {k: round(float(v), 6) for k, v in edge_by_sigma.items()},
        # HW4/HW7
        "green_component_count": int(green_comp_count),
        "green_component_areas_top5": [int(a) for a in green_top5],
        "green_small_component_ratio": round(float(green_small_ratio), 6),
    }
    return out


def compute_path_a_metrics(image_path: str):
    """
    main func for Path A
    output is just a dict of numbers
    """
    bgr = load_bgr(image_path)
    if bgr is None:
        raise FileNotFoundError(f"cant read image: {image_path}")
    return compute_path_a_metrics_from_bgr(bgr)


def save_debug_masks(bgr: np.ndarray, out_base_path: str):
    """
    Save simple debug views:
    - green HSV mask
    - Canny edge map on grayscale
    Files are named:
      out_base_path + "_green_mask.png"
      out_base_path + "_edges.png"
    """
    # green mask
    _, green_mask = green_coverage_hsv(bgr)
    cv2.imwrite(out_base_path + "_green_mask.png", green_mask)

    # edges on gray
    gray = _to_gray_u8(bgr)
    _, edges = edge_density_canny_gray(gray)
    cv2.imwrite(out_base_path + "_edges.png", edges)


def draw_hud_overlay(
    bgr: np.ndarray,
    path_a: dict,
    clip: dict,
    yolo: dict,
    quality_grade: str,
    out_path: str,
):
    """
    Very simple HUD:
    - Translucent sidebar for text
    - YOLO boxes in bright green
    - Telemetry text inside sidebar
    """
    if bgr is None:
        return

    img = bgr.copy()
    h, w = img.shape[:2]

    # ---- translucent sidebar on the left ----
    overlay = img.copy()
    sidebar_width = int(0.32 * w)
    cv2.rectangle(
        overlay,
        (0, 0),
        (sidebar_width, h),
        (0, 0, 0),
        thickness=-1,
    )
    # blend overlay and original
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # ---- YOLO boxes (bright green) ----
    dets = (yolo or {}).get("detections", []) or []
    for det in dets:
        box = det.get("box_xyxy", None)
        label = str(det.get("label", "obj"))
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (pt1[0], max(0, pt1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # ---- telemetry text inside sidebar ----
    scene = str((clip or {}).get("primary_scene", "unknown"))
    subject_summary = "not sure"
    # subject_summary is filled by fusion and passed via report, but in HUD we only see path_a/clip/yolo.
    # To keep things simple and close to your plan, we let the caller pass a subject_summary in path_a if desired.
    subject_summary = str(path_a.get("_subject_summary", "not sure"))

    lap = float(path_a.get("laplacian_variance", 0.0) or 0.0)
    edge_tuned = float(path_a.get("edge_density_tuned", path_a.get("edge_density", 0.0)) or 0.0)
    green_pct = float(path_a.get("green_coverage_percentage", 0.0) or 0.0)

    grade = str(quality_grade or "REVIEW").upper()
    if grade == "PASS":
        grade_color = (0, 255, 0)
    elif grade == "REJECT":
        grade_color = (0, 0, 255)
    else:
        grade_color = (0, 255, 255)  # REVIEW -> yellow

    x0 = 10
    y0 = 30
    dy = 25

    # QUALITY GRADE
    cv2.putText(
        img,
        f"QUALITY: {grade}",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        grade_color,
        2,
        cv2.LINE_AA,
    )
    y0 += dy

    # SCENE
    cv2.putText(
        img,
        f"SCENE: {scene}",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y0 += dy

    # SUBJECT
    cv2.putText(
        img,
        f"SUBJECT: {subject_summary}",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y0 += dy

    # SHARPNESS
    cv2.putText(
        img,
        f"SHARPNESS: {lap:.1f}",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y0 += dy

    # ORGANIC %
    cv2.putText(
        img,
        f"ORGANIC %: {green_pct:.2f}",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # For completeness, also show edge density line
    y0 += dy
    cv2.putText(
        img,
        f"EDGE DENS: {edge_tuned:.3f}",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Quality grade is already shown at top of sidebar
    # Save HUD (uppercase HUD in filename is handled by caller)
    cv2.imwrite(out_path, img)

    grade = str(quality_grade or "REVIEW").upper()
    if grade == "PASS":
        color = (0, 255, 0)
    elif grade == "REJECT":
        color = (0, 0, 255)
    else:
        color = (0, 255, 255)  # REVIEW -> yellow

    grade_text = f"QUALITY: {grade}"
    # simple right align: estimate text width
    (tw, th), _ = cv2.getTextSize(grade_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = max(10, w - tw - 10)
    y = h - 15
    cv2.putText(
        img,
        grade_text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(out_path, img)

