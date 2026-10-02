#!/usr/bin/env python3
"""
GenEval evaluation server for FlowGRPO training.

Uses HuggingFace transformers (Mask2Former + CLIP) — no mmdet/open_clip needed.
Accepts pickle'd requests matching the FlowGRPO geneval_score() interface.

Launch:
    python reward_servers/geneval_server.py --port 18085

The training script connects to http://127.0.0.1:18085 (override with GENEVAL_SERVER_URLS)
when the reward includes "geneval".
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = ["red", "orange", "yellow", "green", "blue", "purple",
          "pink", "brown", "black", "white"]

# COCO "thing" class names used by GenEval (80 classes, order matters)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "computer mouse",
    "tv remote", "computer keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# HF Mask2Former uses Pascal-VOC-style names for some classes that differ from
# the COCO standard names used in GenEval's object_names.txt. This mapping
# translates HF labels → GenEval names so detection results match prompt metadata.
HF_TO_GENEVAL = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "tvmonitor": "tv",
    "mouse": "computer mouse",
    "remote": "tv remote",
    "keyboard": "computer keyboard",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_detector(model_path, device):
    """Load Mask2Former from HuggingFace transformers."""
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    name = "facebook/mask2former-swin-small-coco-instance"
    src = model_path if (model_path and os.path.isdir(model_path)) else name

    print(f"  Loading Mask2Former from {src} ...", flush=True)
    processor = AutoImageProcessor.from_pretrained(src)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(src)
    model = model.to(device).eval()

    # Build label mapping: HF id -> lowercase COCO name
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    # Validate that after HF_TO_GENEVAL mapping, all labels are valid GenEval names
    geneval_names = set(COCO_CLASSES)
    for hf_id, hf_name in id2label.items():
        mapped = HF_TO_GENEVAL.get(hf_name, hf_name)
        if mapped not in geneval_names:
            raise RuntimeError(
                f"HF Mask2Former label id={hf_id} name='{hf_name}' maps to "
                f"'{mapped}' which is not in GenEval COCO_CLASSES. "
                f"Add an entry to HF_TO_GENEVAL in geneval_server.py."
            )

    return model, processor, id2label


def load_clip(device):
    """Load CLIP for color classification."""
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-large-patch14"
    print(f"  Loading CLIP from {name} ...", flush=True)
    model = CLIPModel.from_pretrained(name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor

# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

def detect_objects(image, detector, processor, id2label, device,
                   threshold=0.3, counting_threshold=0.9,
                   max_objects=16, nms_threshold=1.0, is_counting=False,
                   min_mask_area_pixels=256):
    """Run Mask2Former → dict of classname -> [(bbox_5d, mask), ...].

    min_mask_area_pixels: drop detections whose binary mask has fewer than this
    many active pixels. Mask2Former's post_process_instance_segmentation can
    produce sub-pixel-scale "wins" — a query head 95% confident on a single
    pixel passes the score threshold even though the detection is noise.
    Defaulting to 256 (≈16×16) — the floor below which CLIP's color
    classifier also can't reliably read a crop anyway.
    """
    conf = counting_threshold if is_counting else threshold

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detector(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]

    detected = defaultdict(list)
    seg_map = result["segmentation"].cpu().numpy()  # (H, W)

    for seg in result["segments_info"]:
        score = seg["score"]
        if score < conf:
            continue
        label_id = seg["label_id"]
        classname = id2label.get(label_id, f"class_{label_id}")
        classname = HF_TO_GENEVAL.get(classname, classname)  # normalize to GenEval names

        mask = (seg_map == seg["id"])
        ys, xs = np.where(mask)
        if len(ys) < min_mask_area_pixels:
            continue
        bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max(), score],
                        dtype=np.float32)
        detected[classname].append((bbox, mask))

    # Sort by confidence desc, limit count, NMS
    for cls in list(detected.keys()):
        dets = sorted(detected[cls], key=lambda x: -x[0][4])
        dets = dets[:max_objects]
        if nms_threshold < 1.0:
            filtered = []
            for d in dets:
                if all(_iou(d[0], f[0]) < nms_threshold for f in filtered):
                    filtered.append(d)
            dets = filtered
        if dets:
            detected[cls] = dets
        else:
            del detected[cls]

    return dict(detected)

# ---------------------------------------------------------------------------
# Color classification via CLIP zero-shot
# ---------------------------------------------------------------------------

_color_text_cache = {}   # classname -> text_embeds (10, D)


def color_classify(image, bboxes, classname, clip_model, clip_processor, device):
    """Classify the color of each detected crop using CLIP."""
    if classname not in _color_text_cache:
        templates = [f"a photo of a {c} {classname}" for c in COLORS]
        txt = clip_processor(text=templates, return_tensors="pt",
                             padding=True, truncation=True).to(device)
        with torch.no_grad():
            te = clip_model.get_text_features(**txt)
            te = te / te.norm(p=2, dim=-1, keepdim=True)
        _color_text_cache[classname] = te

    text_embeds = _color_text_cache[classname]
    colors = []
    for bbox, mask in bboxes:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Mask-aware crop
        arr = np.array(image)
        if mask is not None:
            bg = np.full_like(arr, 128)
            bg[mask] = arr[mask]
            crop = Image.fromarray(bg[y1:y2, x1:x2])
        else:
            crop = image.crop((x1, y1, x2, y2))
        # Ensure crop is at least 1x1
        if crop.width == 0 or crop.height == 0:
            colors.append("unknown")
            continue
        # Ensure crop is RGB (masking can produce grayscale/single-channel crops)
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        pix = clip_processor(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            ie = clip_model.get_image_features(**pix)
            ie = ie / ie.norm(p=2, dim=-1, keepdim=True)
        sim = (ie @ text_embeds.T).squeeze(0)
        colors.append(COLORS[sim.argmax().item()])
    return colors

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a, b):
    def _area(box):
        return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    inter = _area([max(a[0], b[0]), max(a[1], b[1]),
                   min(a[2], b[2]), min(a[3], b[3])])
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0


def _relative_position(obj_a, obj_b, pos_thresh):
    boxes = np.array([obj_a[0][:4], obj_b[0][:4]]).reshape(2, 2, 2)
    ca, cb = boxes.mean(axis=-2)
    da, db = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = ca - cb
    revised = np.maximum(np.abs(offset) - pos_thresh * (da + db), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / np.linalg.norm(offset)
    rels = set()
    if dx < -0.5: rels.add("left of")
    if dx > 0.5:  rels.add("right of")
    if dy < -0.5: rels.add("above")
    if dy > 0.5:  rels.add("below")
    return rels

# ---------------------------------------------------------------------------
# Evaluation (ported from GenEval evaluate_images.py)
# ---------------------------------------------------------------------------

def evaluate_single(image, objects, metadata,
                    clip_model, clip_processor, device, pos_thresh=0.1):
    """
    Returns (strict_correct: float, partial_score: float).
    strict = 1.0 iff ALL include/exclude clauses pass; else 0.0.
    partial = fraction of clauses that pass.
    """
    total = 0
    passed = 0
    correct = True
    matched_groups = []

    for req in metadata.get("include", []):
        total += 1
        classname = req["class"]
        matched = True
        found = objects.get(classname, [])[:req["count"]]

        if len(found) < req["count"]:
            correct = matched = False
        else:
            if "color" in req:
                cols = color_classify(image, found, classname,
                                      clip_model, clip_processor, device)
                if cols.count(req["color"]) < req["count"]:
                    correct = matched = False
            if "position" in req and matched:
                exp_rel, tgt_grp = req["position"]
                if tgt_grp >= len(matched_groups) or matched_groups[tgt_grp] is None:
                    correct = matched = False
                else:
                    for obj in found:
                        for tgt in matched_groups[tgt_grp]:
                            rels = _relative_position(obj, tgt, pos_thresh)
                            if exp_rel not in rels:
                                correct = matched = False
                                break
                        if not matched:
                            break

        if matched:
            matched_groups.append(found)
            passed += 1
        else:
            matched_groups.append(None)

    for req in metadata.get("exclude", []):
        total += 1
        classname = req["class"]
        if len(objects.get(classname, [])) >= req["count"]:
            correct = False
        else:
            passed += 1

    strict = 1.0 if correct else 0.0
    partial = passed / total if total > 0 else 1.0
    return strict, partial

# ---------------------------------------------------------------------------
# HTTP server (pickle protocol matching FlowGRPO geneval_score)
# ---------------------------------------------------------------------------

class GenevalHandler(BaseHTTPRequestHandler):
    """
    POST: pickle'd {"images": [jpeg_bytes...], "meta_datas": [dict...],
                     "only_strict": bool}
    Response: pickle'd {"scores", "rewards", "strict_rewards",
                        "group_rewards", "group_strict_rewards"}
    """

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        data = pickle.loads(self.rfile.read(length))

        images_bytes = data["images"]
        metadatas = data["meta_datas"]

        scores, rewards, strict_rewards = [], [], []
        group_rewards = defaultdict(list)
        group_strict_rewards = defaultdict(list)

        for img_bytes, meta in zip(images_bytes, metadatas):
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            is_counting = meta.get("tag") == "counting"

            detected = detect_objects(
                img, detector, det_processor, id2label, device,
                threshold=THRESHOLD, counting_threshold=COUNTING_THRESHOLD,
                max_objects=MAX_OBJECTS, nms_threshold=NMS_THRESHOLD,
                is_counting=is_counting,
                min_mask_area_pixels=MIN_MASK_AREA_PIXELS,
            )

            try:
                strict, partial = evaluate_single(
                    img, detected, meta,
                    clip_model, clip_processor, device,
                    pos_thresh=POSITION_THRESHOLD,
                )
            except Exception as e:
                print(f"[GenEval] evaluate_single failed: {e}", flush=True)
                strict, partial = 0.0, 0.0

            tag = meta.get("tag", "unknown")
            scores.append(strict)
            strict_rewards.append(strict)
            rewards.append(partial)
            group_strict_rewards[tag].append(strict)
            group_rewards[tag].append(partial)

        resp = {
            "scores": scores,
            "rewards": rewards,
            "strict_rewards": strict_rewards,
            "group_rewards": dict(group_rewards),
            "group_strict_rewards": dict(group_strict_rewards),
        }
        self.send_response(200)
        self.end_headers()
        self.wfile.write(pickle.dumps(resp))

    def do_GET(self):
        """Health-check endpoint."""
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        # Only log errors, suppress per-request noise
        if args and "200" not in str(args[0]):
            super().log_message(fmt, *args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GenEval evaluation server")
    parser.add_argument("--port", type=int, default=18085)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local path to Mask2Former weights (HF format). "
                             "If omitted, downloads facebook/mask2former-swin-small-coco-instance.")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args()

    global detector, det_processor, id2label
    global clip_model, clip_processor, device
    global THRESHOLD, COUNTING_THRESHOLD, MAX_OBJECTS, NMS_THRESHOLD, POSITION_THRESHOLD, MIN_MASK_AREA_PIXELS

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    THRESHOLD = 0.3
    COUNTING_THRESHOLD = 0.9
    MAX_OBJECTS = 16
    NMS_THRESHOLD = 1.0
    POSITION_THRESHOLD = 0.1
    MIN_MASK_AREA_PIXELS = 256

    t0 = time.time()
    detector, det_processor, id2label = load_detector(args.model_path, device)
    clip_model, clip_processor = load_clip(device)
    print(f"Models loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"GenEval server listening on 0.0.0.0:{args.port}", flush=True)

    server = HTTPServer(("0.0.0.0", args.port), GenevalHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
