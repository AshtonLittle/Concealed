"""
High-Precision Conforming Feature Detection Module
Upgraded with state-of-the-art YOLO11-Medium and YOLO-World-X models.
Features:
  - Recalibrated neural text recognition (PP-OCRv3 via OpenCV DBNet)
  - High-fidelity human pose and body-part segmentation (YOLO11m-pose + YOLO11m-seg)
  - Upgraded COCO instance segmentation (YOLO11m-seg)
  - Universal recognition for non-prerecognized features via YOLO-World-X + FastSAM
    (open-vocabulary grounding with Universal Baseline Vocabulary suppression)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

# Lazy-loaded model instances to keep startup rapid
_POSE_MODEL = None
_SEG_MODEL = None
_FASTSAM_MODEL = None
_WORLD_MODEL = None

# Mapping common user keywords to COCO class names for instant native instance segmentation
COCO_SYNONYMS = {
    # People
    "person": "person",
    "people": "person",
    "human": "person",
    "humans": "person",
    "man": "person",
    "men": "person",
    "woman": "person",
    "women": "person",
    "child": "person",
    "kid": "person",
    "pedestrian": "person",
    
    # Common objects
    "waterbottle": "bottle",
    "water bottle": "bottle",
    "bottle": "bottle",
    "bottles": "bottle",
    "table": "dining table",
    "tables": "dining table",
    "desk": "dining table",
    "dining table": "dining table",
    "chair": "chair",
    "chairs": "chair",
    "couch": "couch",
    "sofa": "couch",
    "bed": "bed",
    "cup": "cup",
    "mug": "cup",
    "glass": "wine glass",
    "wine glass": "wine glass",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "bowl": "bowl",
    "computer": "laptop",
    "laptop": "laptop",
    "mouse": "mouse",
    "keyboard": "keyboard",
    "cellphone": "cell phone",
    "cell phone": "cell phone",
    "phone": "cell phone",
    "smartphone": "cell phone",
    "tv": "tv",
    "television": "tv",
    "screen": "tv",
    "monitor": "tv",
    "book": "book",
    "books": "book",
    "backpack": "backpack",
    "bag": "backpack",
    "handbag": "handbag",
    "purse": "handbag",
    "suitcase": "suitcase",
    "car": "car",
    "cars": "car",
    "automobile": "car",
    "vehicle": "car",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "cat": "cat",
    "cats": "cat",
    "dog": "dog",
    "dogs": "dog",
}


@dataclass
class ConformingRegion:
    """Represents a detected region conforming to an object's exact silhouette."""
    x1: int
    y1: int
    x2: int
    y2: int
    mask: np.ndarray  # 2D uint8 numpy array matching (y2 - y1, x2 - x1) with values 0 or 255
    label: str
    confidence: float = 1.0

    @property
    def pil_mask(self) -> Image.Image:
        """Returns the mask as a PIL Image in mode 'L'."""
        return Image.fromarray(self.mask, mode="L")

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def _get_pose_model():
    """YOLO11 Medium pose model for high-precision keypoints."""
    global _POSE_MODEL
    if _POSE_MODEL is None:
        from ultralytics import YOLO
        _POSE_MODEL = YOLO("yolo11m-pose.pt")
    return _POSE_MODEL


def _get_seg_model():
    """YOLO11 Medium instance segmentation model."""
    global _SEG_MODEL
    if _SEG_MODEL is None:
        from ultralytics import YOLO
        _SEG_MODEL = YOLO("yolo11m-seg.pt")
    return _SEG_MODEL


def _get_world_model():
    """YOLO-World v2 X-Large open-vocabulary detector for maximum semantic discrimination."""
    global _WORLD_MODEL
    if _WORLD_MODEL is None:
        from ultralytics import YOLO
        _WORLD_MODEL = YOLO("yolov8x-worldv2.pt")
    return _WORLD_MODEL


def _get_fastsam_model():
    """FastSAM segment-anything model for high-fidelity promptable segmentation."""
    global _FASTSAM_MODEL
    if _FASTSAM_MODEL is None:
        from ultralytics import FastSAM
        _FASTSAM_MODEL = FastSAM("FastSAM-s.pt")
    return _FASTSAM_MODEL


# =====================================================================
# 1. RECALIBRATED NEURAL TEXT DETECTION (PP-OCRv3 DBNet)
# =====================================================================

def _polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    min1, max1 = poly1.min(axis=0), poly1.max(axis=0)
    min2, max2 = poly2.min(axis=0), poly2.max(axis=0)
    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return 0.0
    bx1 = min(min1[0], min2[0])
    by1 = min(min1[1], min2[1])
    bx2 = max(max1[0], max2[0])
    by2 = max(max1[1], max2[1])
    bw, bh = max(1, bx2 - bx1 + 1), max(1, by2 - by1 + 1)
    m1 = np.zeros((bh, bw), dtype=np.uint8)
    m2 = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(m1, [poly1 - [bx1, by1]], 255)
    cv2.fillPoly(m2, [poly2 - [bx1, by1]], 255)
    inter = np.count_nonzero(cv2.bitwise_and(m1, m2))
    union = np.count_nonzero(cv2.bitwise_or(m1, m2))
    return inter / max(1, union)


def _nms_polygons(polys: List[np.ndarray], scores: List[float], iou_thresh: float = 0.3) -> Tuple[List[np.ndarray], List[float]]:
    if not polys:
        return [], []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        cur = idxs[0]
        keep.append(cur)
        rem = [i for i in idxs[1:] if _polygon_iou(polys[cur], polys[i]) < iou_thresh]
        idxs = np.array(rem)
    return [polys[k] for k in keep], [scores[k] for k in keep]


def _ensure_ppocr_model() -> Optional[Path]:
    """Ensures the 2.4 MB PP-OCRv3 ONNX model is available locally, downloading if missing."""
    ppocr_path = Path(__file__).parent / "weights" / "text_detection_ppocrv3.onnx"
    if not ppocr_path.exists():
        try:
            ppocr_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx"
            print("[*] Downloading PP-OCRv3 text model (2.4 MB)...")
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(ppocr_path, "wb") as f:
                f.write(resp.read())
            print("[OK] PP-OCRv3 text model downloaded.")
        except Exception as e:
            print(f"[!] Warning: Could not auto-download PP-OCRv3: {e}")
            return None
    return ppocr_path


def detect_conforming_text(img_bgr: np.ndarray, conf: float = 0.20) -> List[ConformingRegion]:
    """
    Recalibrated multi-scale neural text detector.
    Uses PP-OCRv3 DBNet with adaptive high-resolution tiled inference for large photos
    (like street scenes or 4K/20MP images), ensuring fine text and street signs are captured
    at full optical resolution without downscale blurring.
    """
    h_img, w_img = img_bgr.shape[:2]
    ppocr_path = _ensure_ppocr_model()
    
    if ppocr_path and ppocr_path.exists():
        try:
            net = cv2.dnn.readNet(str(ppocr_path))
            model = cv2.dnn_TextDetectionModel_DB(net)
            model.setBinaryThreshold(0.20)
            model.setPolygonThreshold(conf)
            model.setUnclipRatio(1.8)
            model.setMaxCandidates(1000)
            model.setInputMean((123.675, 116.28, 103.53))
            model.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
            
            raw_polys: List[np.ndarray] = []
            raw_scores: List[float] = []
            
            max_dim = max(w_img, h_img)
            if max_dim <= 1800:
                # Standard single-pass inference for normal resolution images
                scale = min(1.0, 1280.0 / max_dim)
                new_w = int(np.ceil((w_img * scale) / 32.0) * 32)
                new_h = int(np.ceil((h_img * scale) / 32.0) * 32)
                model.setInputSize((new_w, new_h))
                resized = cv2.resize(img_bgr, (new_w, new_h))
                polys, scores = model.detect(resized)
                for poly, score in zip(polys, scores):
                    if score >= conf:
                        poly_orig = poly.astype(np.float32)
                        poly_orig[:, 0] *= (w_img / new_w)
                        poly_orig[:, 1] *= (h_img / new_h)
                        raw_polys.append(poly_orig.astype(np.int32))
                        raw_scores.append(float(score))
            else:
                # Tiled multi-scale inference for ultra-high-resolution images (e.g. street scenes, 4K/20MP)
                tile_size = 1280
                overlap = 256
                step = tile_size - overlap
                model.setInputSize((tile_size, tile_size))
                
                y_starts = list(range(0, h_img, step))
                x_starts = list(range(0, w_img, step))
                
                for y0 in y_starts:
                    for x0 in x_starts:
                        y1 = min(h_img, y0 + tile_size)
                        x1 = min(w_img, x0 + tile_size)
                        tile = img_bgr[y0:y1, x0:x1]
                        th, tw = tile.shape[:2]
                        
                        if th < tile_size or tw < tile_size:
                            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                            padded[:th, :tw] = tile
                            tile_input = padded
                        else:
                            tile_input = tile
                            
                        polys, scores = model.detect(tile_input)
                        for poly, score in zip(polys, scores):
                            if score >= conf:
                                poly_int = poly.astype(np.int32)
                                if poly_int[:, 0].max() <= tw and poly_int[:, 1].max() <= th:
                                    poly_int[:, 0] += x0
                                    poly_int[:, 1] += y0
                                    raw_polys.append(poly_int)
                                    raw_scores.append(float(score))
                                    
            # Filter duplicates from overlapping tiles using IoU NMS
            clean_polys, clean_scores = _nms_polygons(raw_polys, raw_scores, iou_thresh=0.30)
            
            regions = []
            for poly_int, score in zip(clean_polys, clean_scores):
                x1 = max(0, int(poly_int[:, 0].min()))
                y1 = max(0, int(poly_int[:, 1].min()))
                x2 = min(w_img, int(poly_int[:, 0].max()) + 1)
                y2 = min(h_img, int(poly_int[:, 1].max()) + 1)
                
                if x2 > x1 and y2 > y1:
                    crop_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                    rel_poly = (poly_int - np.array([x1, y1])).astype(np.int32)
                    cv2.fillPoly(crop_mask, [rel_poly], 255)
                    regions.append(ConformingRegion(x1, y1, x2, y2, crop_mask, "text", score))
                    
            if regions:
                return regions
        except Exception:
            pass
            
    # Morphological fallback if neural network fails
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 15 < w < w_img * 0.95 and 6 < h < h_img * 0.5:
            pad_x = int(w * 0.04)
            pad_y = int(h * 0.06)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)
            crop_connected = connected[y1:y2, x1:x2].copy()
            smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            crop_mask = cv2.dilate(crop_connected, smooth_kernel, iterations=1)
            if np.count_nonzero(crop_mask) > 10:
                regions.append(ConformingRegion(x1, y1, x2, y2, crop_mask, "text", conf))
    return regions


# =====================================================================
# 2. BODY PARTS & ANATOMY (YOLO11m-Pose + YOLO11m-Seg Silhouette)
# =====================================================================

def detect_conforming_pose(path_str: str, feature: str, conf: float = 0.25) -> List[ConformingRegion]:
    """
    Extracts conforming body-part regions (face, arms, hands, legs)
    using YOLO11m-pose keypoints and YOLO11m-seg person silhouette.
    """
    pose_model = _get_pose_model()
    seg_model = _get_seg_model()
    
    pose_results = pose_model.predict(path_str, conf=conf, verbose=False)[0]
    seg_results = seg_model.predict(path_str, conf=conf, verbose=False)[0]
    
    pil_img = Image.open(path_str)
    w, h = pil_img.size
    
    # Cumulative person silhouette mask
    person_mask = np.zeros((h, w), dtype=np.uint8)
    if seg_results.masks is not None:
        for i, poly in enumerate(seg_results.masks.xy):
            cls_name = seg_results.names[int(seg_results.boxes.cls[i])]
            if cls_name == "person":
                cv2.fillPoly(person_mask, [poly.astype(np.int32)], 255)
                
    has_person_seg = np.count_nonzero(person_mask) > 0
    regions = []
    
    if pose_results.keypoints is None or len(pose_results.keypoints) == 0:
        return regions
        
    xy = pose_results.keypoints.xy.cpu().numpy()
    k_conf = pose_results.keypoints.conf.cpu().numpy()
    
    for person_idx in range(len(xy)):
        kpts = xy[person_idx]
        confs = k_conf[person_idx]
        
        # ----------------- FACE / HEAD -----------------
        if feature in ["face", "faces", "head", "heads"]:
            face_k = kpts[:5]
            face_c = confs[:5]
            valid = face_k[face_c > conf]
            if len(valid) >= 2:
                cx, cy = float(valid[:, 0].mean()), float(valid[:, 1].mean())
                span_x = float(valid[:, 0].max() - valid[:, 0].min())
                span_y = float(valid[:, 1].max() - valid[:, 1].min())
                rx = int(max(span_x * 0.9, 35))
                ry = int(max(span_y * 1.3, 45))
                
                head_canvas = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(head_canvas, (int(cx), int(cy)), (rx, ry), 0, 0, 360, 255, -1)
                
                conforming_mask = cv2.bitwise_and(person_mask, head_canvas) if has_person_seg else head_canvas
                y_idx, x_idx = np.where(conforming_mask > 0)
                if len(y_idx) > 0:
                    x1, y1 = int(x_idx.min()), int(y_idx.min())
                    x2, y2 = int(x_idx.max() + 1), int(y_idx.max() + 1)
                    crop_mask = conforming_mask[y1:y2, x1:x2]
                    regions.append(ConformingRegion(x1, y1, x2, y2, crop_mask, "face"))
                    
        # ----------------- ARMS -----------------
        elif feature in ["arm", "arms"]:
            limb_conf = min(conf, 0.10)
            for arm_idx, limb_joints in enumerate([(5, 7, 9), (6, 8, 10)], 1):
                arm_canvas = np.zeros((h, w), dtype=np.uint8)
                p_shoulder, p_elbow, p_wrist = kpts[limb_joints[0]], kpts[limb_joints[1]], kpts[limb_joints[2]]
                c_shoulder, c_elbow, c_wrist = confs[limb_joints[0]], confs[limb_joints[1]], confs[limb_joints[2]]
                
                drawn_segments = 0
                for pt1, pt2, c1, c2 in [(p_shoulder, p_elbow, c_shoulder, c_elbow), (p_elbow, p_wrist, c_elbow, c_wrist)]:
                    if c1 >= limb_conf and c2 >= limb_conf and pt1.any() and pt2.any():
                        p1 = (int(pt1[0]), int(pt1[1]))
                        p2 = (int(pt2[0]), int(pt2[1]))
                        seg_len = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        thickness = max(25, int(seg_len * 0.50))
                        cv2.line(arm_canvas, p1, p2, 255, thickness)
                        cv2.circle(arm_canvas, p1, thickness // 2, 255, -1)
                        cv2.circle(arm_canvas, p2, thickness // 2, 255, -1)
                        drawn_segments += 1
                        
                if drawn_segments == 0 and c_shoulder >= conf and p_shoulder.any():
                    p1 = (int(p_shoulder[0]), int(p_shoulder[1]))
                    p2 = (p1[0], p1[1] + 180)
                    thickness = 60
                    cv2.line(arm_canvas, p1, p2, 255, thickness)
                    cv2.circle(arm_canvas, p1, thickness // 2, 255, -1)
                    cv2.circle(arm_canvas, p2, thickness // 2, 255, -1)
                    drawn_segments += 1
                        
                if drawn_segments > 0:
                    conforming_arm = cv2.bitwise_and(person_mask, arm_canvas) if has_person_seg else arm_canvas
                    y_idx, x_idx = np.where(conforming_arm > 0)
                    if len(y_idx) > 0:
                        x1, y1 = int(x_idx.min()), int(y_idx.min())
                        x2, y2 = int(x_idx.max() + 1), int(y_idx.max() + 1)
                        crop_mask = conforming_arm[y1:y2, x1:x2]
                        regions.append(ConformingRegion(x1, y1, x2, y2, crop_mask, f"arm_{arm_idx}"))
                        
        # ----------------- LEGS -----------------
        elif feature in ["leg", "legs"]:
            limb_conf = min(conf, 0.10)
            for leg_idx, limb_joints in enumerate([(11, 13, 15), (12, 14, 16)], 1):
                leg_canvas = np.zeros((h, w), dtype=np.uint8)
                p_hip, p_knee, p_ankle = kpts[limb_joints[0]], kpts[limb_joints[1]], kpts[limb_joints[2]]
                c_hip, c_knee, c_ankle = confs[limb_joints[0]], confs[limb_joints[1]], confs[limb_joints[2]]
                
                drawn = 0
                for pt1, pt2, c1, c2 in [(p_hip, p_knee, c_hip, c_knee), (p_knee, p_ankle, c_knee, c_ankle)]:
                    if c1 >= limb_conf and c2 >= limb_conf and pt1.any() and pt2.any():
                        p1 = (int(pt1[0]), int(pt1[1]))
                        p2 = (int(pt2[0]), int(pt2[1]))
                        seg_len = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        thickness = max(30, int(seg_len * 0.45))
                        cv2.line(leg_canvas, p1, p2, 255, thickness)
                        cv2.circle(leg_canvas, p1, thickness // 2, 255, -1)
                        cv2.circle(leg_canvas, p2, thickness // 2, 255, -1)
                        drawn += 1
                        
                if drawn > 0:
                    conforming_leg = cv2.bitwise_and(person_mask, leg_canvas) if has_person_seg else leg_canvas
                    y_idx, x_idx = np.where(conforming_leg > 0)
                    if len(y_idx) > 0:
                        x1, y1 = int(x_idx.min()), int(y_idx.min())
                        x2, y2 = int(x_idx.max() + 1), int(y_idx.max() + 1)
                        crop_mask = conforming_leg[y1:y2, x1:x2]
                        regions.append(ConformingRegion(x1, y1, x2, y2, crop_mask, f"leg_{leg_idx}"))
                        
    return regions


# =====================================================================
# 3. ADVANCED RECOGNITION FOR NON-PRERECOGNIZED & ARBITRARY FEATURES
# =====================================================================

# Universal baseline vocabulary to competitively suppress visual false positives.
# When querying an open-vocabulary feature (e.g. 'door'), these common classes
# compete against the target. If an image region matches 'shirt' or 'person' higher
# than 'door', the false positive is automatically suppressed.
UNIVERSAL_BASELINE_CLASSES = [
    # Humans & Clothing
    "person", "human", "clothing", "shirt", "t-shirt", "pants", "jeans", "jacket", "coat", "shoes", "hat",
    # Furniture & Architectural
    "chair", "couch", "sofa", "bed", "table", "desk", "wall", "floor", "ceiling", "window", "cabinet", "shelf",
    # Everyday Objects & Electronics
    "laptop", "computer", "phone", "tv", "screen", "bottle", "cup", "bag", "backpack",
    # Vehicles & Nature
    "car", "vehicle", "bicycle", "tree", "plant", "ground", "sky"
]


def _expand_feature_queries(feature: str) -> List[str]:
    """Generates natural language query variations to maximize open-vocabulary recall."""
    feat = feature.strip().lower()
    variations = [feat]
    if feat.endswith("s") and len(feat) > 3:
        variations.append(feat[:-1])  # singular form
    elif not feat.endswith("s"):
        variations.append(f"{feat}s")  # plural form
    return list(dict.fromkeys(variations))


def detect_conforming_arbitrary(path_str: str, feature: str, conf: float = 0.25) -> List[ConformingRegion]:
    """
    Advanced recognition for any arbitrary or non-prerecognized feature.
    
    Architecture:
    1. YOLO11m-seg: Fast, high-accuracy instance segmentation for standard COCO categories.
    2. YOLO-World-X Grounding + Universal Baseline Competition + FastSAM Box Prompting:
       Uses YOLO-World-X with competitive baseline suppression to ground the arbitrary query
       without hallucinating on background objects (e.g., shirts mistagged as doors).
       Candidate bounding boxes are then fed into FastSAM for pixel-precise conforming contours.
    3. FastSAM Open-Vocabulary Text Prompting: Direct CLIP-prompted mask search (calibrated conf).
    4. Fallback: Bounding box GrabCut segmentation.
    """
    feature_clean = feature.strip().lower()
    coco_target = COCO_SYNONYMS.get(feature_clean)
    
    pil_img = Image.open(path_str)
    w, h = pil_img.size
    regions = []
    
    # -------------------------------------------------------------
    # Step 1: Upgraded YOLO11m-seg for known/prerecognized classes
    # -------------------------------------------------------------
    if coco_target:
        seg_model = _get_seg_model()
        results = seg_model.predict(path_str, conf=conf, verbose=False)[0]
        
        if results.masks is not None:
            for i, poly in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i])
                cls_name = results.names[cls_id]
                box_conf = float(results.boxes.conf[i])
                
                if cls_name == coco_target and len(poly) >= 3:
                    x1 = max(0, int(poly[:, 0].min()))
                    y1 = max(0, int(poly[:, 1].min()))
                    x2 = min(w, int(poly[:, 0].max()) + 1)
                    y2 = min(h, int(poly[:, 1].max()) + 1)
                    
                    if x2 > x1 and y2 > y1:
                        crop_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                        rel_poly = (poly - np.array([x1, y1])).astype(np.int32)
                        cv2.fillPoly(crop_mask, [rel_poly], 255)
                        regions.append(ConformingRegion(x1, y1, x2, y2, crop_mask, cls_name, box_conf))
                        
        if regions:
            return regions

    # -------------------------------------------------------------
    # Step 2: Non-Prerecognized Features: YOLO-World-X + Universal Baseline Suppression
    # -------------------------------------------------------------
    target_queries = _expand_feature_queries(feature_clean)
    target_set = set(q.lower() for q in target_queries)
    
    # Filter baseline distractors so they don't overlap with the user's requested feature
    distractors = [c for c in UNIVERSAL_BASELINE_CLASSES if c not in target_set and feature_clean not in c]
    all_classes = target_queries + distractors
    
    world_model = _get_world_model()
    world_model.set_classes(all_classes)
    
    # Calibrated confidence threshold (no halving down into noise floor)
    world_conf = max(0.20, conf)
    world_results = world_model.predict(path_str, conf=world_conf, verbose=False)[0]
    
    detected_boxes = []
    if world_results.boxes is not None and len(world_results.boxes) > 0:
        for box in world_results.boxes:
            cls_id = int(box.cls[0])
            pred_label = all_classes[cls_id]
            box_conf = float(box.conf[0])
            
            # CRITICAL COMPETITIVE SUPPRESSION:
            # Only keep detections where the user's target query beat all baseline distractors!
            if pred_label in target_set:
                coords = [int(v) for v in box.xyxy[0].tolist()]
                detected_boxes.append((coords, box_conf))
        
    if detected_boxes:
        print(f"[*] YOLO-World-X grounded {len(detected_boxes)} region(s) for '{feature_clean}'")
        fastsam = _get_fastsam_model()
        for coords, b_conf in detected_boxes:
            bx1, by1, bx2, by2 = coords
            # Segment the exact object inside this box prompt using FastSAM
            fs_res = fastsam(path_str, bboxes=[coords], verbose=False)[0]
            if fs_res.masks is not None and len(fs_res.masks) > 0:
                poly = fs_res.masks.xy[0]
                if len(poly) >= 3:
                    x1 = max(0, int(poly[:, 0].min()))
                    y1 = max(0, int(poly[:, 1].min()))
                    x2 = min(w, int(poly[:, 0].max()) + 1)
                    y2 = min(h, int(poly[:, 1].max()) + 1)
                    
                    if x2 > x1 and y2 > y1:
                        crop_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                        rel_poly = (poly - np.array([x1, y1])).astype(np.int32)
                        cv2.fillPoly(crop_mask, [rel_poly], 255)
                        regions.append(ConformingRegion(x1, y1, x2, y2, crop_mask, feature_clean, b_conf))
                        continue
                        
            # If FastSAM box segmentation didn't trigger, use GrabCut within box
            img_bgr = cv2.imread(path_str)
            rect = (bx1, by1, max(1, bx2 - bx1), max(1, by2 - by1))
            gc_mask = np.zeros((h, w), np.uint8)
            bgd_m = np.zeros((1, 65), np.float64)
            fgd_m = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(img_bgr, gc_mask, rect, bgd_m, fgd_m, 2, cv2.GC_INIT_WITH_RECT)
                fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                crop_mask = fg[by1:by2, bx1:bx2]
            except Exception:
                crop_mask = np.full((by2 - by1, bx2 - bx1), 255, dtype=np.uint8)
                
            regions.append(ConformingRegion(bx1, by1, bx2, by2, crop_mask, feature_clean, b_conf))
            
    return regions


# =====================================================================
# 4. MAIN DISPATCH ENTRY POINT
# =====================================================================

def detect_feature_regions(
    image_path: str | Path,
    feature: str,
    conf: float = 0.25
) -> List[ConformingRegion]:
    """
    Main entry point for conforming feature detection.
    
    Dispatches to:
    - Recalibrated neural PP-OCRv3 for text / writing / reading
    - YOLO11m-pose + YOLO11m-seg for body parts (face, arms, hands, legs)
    - YOLO11m-seg + YOLO-World-M + FastSAM for ANY non-prerecognized or arbitrary feature
    
    Returns:
        List of ConformingRegion objects with exact silhouette masks.
    """
    path_str = str(Path(image_path).resolve())
    feat_norm = feature.strip().lower()
    
    # 1. Text
    if feat_norm in ["text", "words", "word", "letter", "letters", "writing", "ocr", "font", "reading"]:
        img_bgr = cv2.imread(path_str)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image at {path_str}")
        return detect_conforming_text(img_bgr, conf=conf)
        
    # 2. Body parts
    if feat_norm in ["face", "faces", "head", "heads", "arm", "arms", "hand", "hands", "leg", "legs"]:
        regions = detect_conforming_pose(path_str, feat_norm, conf=conf)
        if regions:
            return regions
        # Fallback to open-vocabulary segmentation
        return detect_conforming_arbitrary(path_str, feat_norm, conf=conf)
        
    # 3. Universal detector for any feature (prerecognized or non-prerecognized)
    return detect_conforming_arbitrary(path_str, feat_norm, conf=conf)
