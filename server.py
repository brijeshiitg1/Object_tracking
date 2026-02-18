"""
Object Detection & Tracking Web Dashboard - Flask Server
Integrates with existing engine (ObjectDetection, MultiObjectTracker, AOIAnalyzer)
"""

import sys, os, time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, send_file, Response
import cv2
import numpy as np
import uuid, threading, subprocess, traceback
from pathlib import Path
from collections import defaultdict

from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracker, KalmanBoxTracker
from engine.aoi_utils import AOIAnalyzer

app = Flask(__name__, template_folder=str(Path(BASE_DIR) / "templates"))

# ── Folders ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = Path(BASE_DIR) / "uploads_web"
OUTPUT_FOLDER = Path(BASE_DIR) / "outputs_web"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = str(Path(BASE_DIR) / "models" / "yolo26n.pt")
print(f"\n🔍 Loading model from {MODEL_PATH} ...")
od = ObjectDetection(model_path=MODEL_PATH)
print("✅ Model loaded!\n")

# ── Target classes ─────────────────────────────────────────────────────────────
TARGET_CLASS_IDS = {0, 1, 2, 3, 5, 7}
CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}
# BGR per class
CLASS_COLORS = {
    0: (219, 112,  70),   # person      – warm blue
    1: (219, 112, 219),   # bicycle     – purple
    2: ( 86, 214,  86),   # car         – green
    3: (219,  86, 214),   # motorcycle  – violet
    5: ( 86, 214, 214),   # bus         – cyan
    7: ( 86, 160, 219),   # truck       – orange-blue
}
# Color used for objects that are inside the AOI
IN_AOI_COLOR = (30, 80, 240)   # bright red-orange (BGR)
# AOI polygon fill/border color (indigo, BGR)
AOI_BORDER_COLOR = (255, 130, 80)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# ── State ──────────────────────────────────────────────────────────────────────
jobs: dict        = {}
jobs_lock         = threading.Lock()
job_frames: dict  = {}           # job_id -> latest JPEG bytes for live stream
frames_lock       = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def update_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def push_frame(job_id: str, frame: np.ndarray, max_width: int = 960):
    """Downscale if needed, JPEG-encode, and store for MJPEG streaming."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    with frames_lock:
        job_frames[job_id] = jpeg.tobytes()


def generate_mjpeg(job_id: str):
    """Yield MJPEG frames until the job finishes or client disconnects."""
    BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    last_data = None
    try:
        while True:
            with jobs_lock:
                job = jobs.get(job_id, {})
            status = job.get("status", "processing")
            with frames_lock:
                frame_data = job_frames.get(job_id)
            if frame_data and frame_data is not last_data:
                last_data = frame_data
                yield BOUNDARY + frame_data + b"\r\n"
            if status in ("done", "error"):
                break
            time.sleep(0.033)
    except GeneratorExit:
        pass


def draw_box(frame, x1, y1, x2, y2, label, color, thickness=1):
    """Draw a thin bounding box with compact label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 7), (x1 + lw + 5, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (10, 10, 10), 1, cv2.LINE_AA)


def draw_aoi_overlay(frame: np.ndarray, pixel_points: list, label: str = "AOI") -> np.ndarray:
    """Draw a semi-transparent AOI polygon with border and centroid label."""
    pts = np.array(pixel_points, dtype=np.int32)
    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], AOI_BORDER_COLOR)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    # Solid border
    cv2.polylines(frame, [pts], isClosed=True, color=AOI_BORDER_COLOR, thickness=2)
    # Label at centroid
    cx = int(np.mean([p[0] for p in pixel_points]))
    cy = int(np.mean([p[1] for p in pixel_points]))
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(frame, (cx - lw//2 - 5, cy - lh - 6), (cx + lw//2 + 5, cy + 3),
                  (25, 18, 55), -1)
    cv2.putText(frame, label, (cx - lw//2, cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 140), 1, cv2.LINE_AA)
    return frame


def draw_stats_overlay(frame: np.ndarray, counts: dict, aoi_counts: dict = None) -> np.ndarray:
    """Draw semi-transparent stats panel at top-left. Shows 'aoi/total' when AOI active."""
    items = [(k, v) for k, v in counts.items() if v > 0]
    if not items:
        return frame
    pad    = 10
    line_h = 26
    box_w  = 200
    box_h  = len(items) * line_h + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (box_w, box_h + pad), (12, 12, 22), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    y = pad * 2 + line_h // 2
    for cls_name, total in items:
        cls_id = next((k for k, v in CLASS_NAMES.items() if v == cls_name), 0)
        color  = CLASS_COLORS.get(cls_id, (255, 255, 255))
        if aoi_counts is not None:
            aoi_c = aoi_counts.get(cls_name, 0)
            label = f"{cls_name}: {aoi_c}"
        else:
            label = f"{cls_name}: {total}"
        cv2.putText(frame, label, (pad * 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        y += line_h
    return frame


def denormalize_aoi(aoi_points: list, width: int, height: int) -> list:
    """Convert normalized [0–1] AOI points to pixel coordinates."""
    return [[int(nx * width), int(ny * height)] for nx, ny in aoi_points]


# ── Processing ─────────────────────────────────────────────────────────────────

def process_image(job_id: str, filepath: str, aoi_points=None):
    try:
        frame = cv2.imread(filepath)
        if frame is None:
            update_job(job_id, status="error", error="Could not read image file")
            return

        h, w = frame.shape[:2]

        # Build AOI polygon (pixel coords) if provided
        pixel_aoi   = None
        aoi_poly_np = None
        if aoi_points and len(aoi_points) >= 3:
            pixel_aoi   = denormalize_aoi(aoi_points, w, h)
            aoi_poly_np = np.array(pixel_aoi, dtype=np.int32)

        # Draw AOI before boxes so boxes render on top
        if pixel_aoi:
            frame = draw_aoi_overlay(frame, pixel_aoi, "Detection Zone")

        bboxes, class_ids, scores = od.detect(frame, img_size=640, conf=0.45)

        counts:     dict = defaultdict(int)
        aoi_counts: dict = defaultdict(int)

        for bbox, cls_id, score in zip(bboxes, class_ids, scores):
            if cls_id not in TARGET_CLASS_IDS:
                continue
            cls_name = CLASS_NAMES[cls_id]
            counts[cls_name] += 1
            x1, y1, x2, y2 = map(int, bbox)
            base_color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            # AOI check using bottom-center point
            in_aoi = False
            if aoi_poly_np is not None:
                cx, cy = (x1 + x2) // 2, y2
                in_aoi = cv2.pointPolygonTest(aoi_poly_np, (float(cx), float(cy)), False) >= 0
                if in_aoi:
                    aoi_counts[cls_name] += 1

            color = IN_AOI_COLOR if in_aoi else base_color
            thick = 2 if in_aoi else 1
            label = f"{cls_name} {score:.2f}" + (" [Z]" if in_aoi else "")
            draw_box(frame, x1, y1, x2, y2, label, color, thickness=thick)

        final_counts = dict(counts)
        final_aoi    = dict(aoi_counts)
        frame = draw_stats_overlay(frame, final_counts, final_aoi if pixel_aoi else None)
        push_frame(job_id, frame)

        output_path = OUTPUT_FOLDER / f"{job_id}_output.jpg"
        cv2.imwrite(str(output_path), frame)

        update_job(job_id,
                   status="done", progress=100,
                   stats=final_counts,   total=sum(final_counts.values()),
                   aoi_stats=final_aoi,  has_aoi=(pixel_aoi is not None),
                   output_file=f"{job_id}_output.jpg", file_type="image")

    except Exception:
        update_job(job_id, status="error", error=traceback.format_exc())


def process_video(job_id: str, filepath: str, aoi_points=None):
    try:
        KalmanBoxTracker.count = 0
        mot     = MultiObjectTracker()
        tracker = mot.ocsort(max_age=30, min_hits=3, iou_threshold=0.3)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            update_job(job_id, status="error", error="Cannot open video file")
            return

        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_out  = OUTPUT_FOLDER / f"{job_id}_temp.mp4"
        final_out = OUTPUT_FOLDER / f"{job_id}_output.mp4"
        writer    = cv2.VideoWriter(str(temp_out),
                                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        # Build AOI
        pixel_aoi    = None
        aoi_analyzer = None
        if aoi_points and len(aoi_points) >= 3:
            pixel_aoi    = denormalize_aoi(aoi_points, width, height)
            aoi_analyzer = AOIAnalyzer([pixel_aoi], ["Detection Zone"])

        unique_ids:    dict = defaultdict(set)   # all IDs ever seen
        aoi_entry_ids: dict = defaultdict(set)   # IDs that entered AOI at least once

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bboxes, class_ids, scores = od.detect(frame, img_size=640, conf=0.45)

            f_bboxes, f_scores, f_cls = [], [], []
            for bbox, cls_id, score in zip(bboxes, class_ids, scores):
                if cls_id in TARGET_CLASS_IDS:
                    f_bboxes.append(bbox)
                    f_scores.append(score)
                    f_cls.append(cls_id)

            tracked = tracker.update(f_bboxes, f_scores, f_cls, frame)

            # Draw AOI polygon (below boxes)
            if pixel_aoi:
                frame = draw_aoi_overlay(frame, pixel_aoi, "Detection Zone")

            for track in tracked:
                x1, y1, x2, y2, obj_id, cls_id, score = track.astype(int)
                cls_name   = CLASS_NAMES.get(int(cls_id), "unknown")
                base_color = CLASS_COLORS.get(int(cls_id), (255, 255, 255))
                obj_id     = int(obj_id)

                unique_ids[cls_name].add(obj_id)

                # AOI check
                in_aoi = False
                if aoi_analyzer:
                    res    = aoi_analyzer.check_object_in_aoi(
                                 [x1, y1, x2, y2], obj_id, use_bottom_center=True)
                    in_aoi = res["in_aoi"]
                    if in_aoi:
                        aoi_entry_ids[cls_name].add(obj_id)

                color = IN_AOI_COLOR if in_aoi else base_color
                thick = 2 if in_aoi else 1
                label = f"ID:{obj_id} {cls_name}" + (" [Z]" if in_aoi else "")
                draw_box(frame, x1, y1, x2, y2, label, color, thickness=thick)

                # Tracking dot at bottom-center
                cx, cy = (x1 + x2) // 2, y2
                cv2.circle(frame, (cx, cy), 3, color, -1)

            cumulative     = {cls: len(ids) for cls, ids in unique_ids.items()}
            aoi_cumulative = {cls: len(ids) for cls, ids in aoi_entry_ids.items()} \
                             if aoi_analyzer else {}

            frame = draw_stats_overlay(frame, cumulative,
                                       aoi_cumulative if aoi_analyzer else None)
            push_frame(job_id, frame)
            writer.write(frame)

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            update_job(job_id,
                       progress=progress,
                       stats=cumulative,      total=sum(cumulative.values()),
                       aoi_stats=aoi_cumulative, has_aoi=(aoi_analyzer is not None))

        cap.release()
        writer.release()

        # Re-encode with ffmpeg for H.264 browser compatibility
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(temp_out),
                 "-vcodec", "libx264", "-crf", "23",
                 "-movflags", "+faststart", "-an", str(final_out)],
                capture_output=True, timeout=600)
            if result.returncode == 0:
                temp_out.unlink(missing_ok=True)
            else:
                temp_out.rename(final_out)
        except Exception:
            try:
                temp_out.rename(final_out)
            except Exception:
                pass

        final_stats     = {cls: len(ids) for cls, ids in unique_ids.items()}
        final_aoi_stats = {cls: len(ids) for cls, ids in aoi_entry_ids.items()} \
                          if aoi_analyzer else {}

        update_job(job_id,
                   status="done", progress=100,
                   stats=final_stats,      total=sum(final_stats.values()),
                   aoi_stats=final_aoi_stats, has_aoi=(aoi_analyzer is not None),
                   output_file=f"{job_id}_output.mp4", file_type="video")

    except Exception:
        update_job(job_id, status="error", error=traceback.format_exc())


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(str(Path(BASE_DIR) / "templates" / "dashboard.html"))


@app.route("/upload", methods=["POST"])
def upload():
    """Save uploaded file and return job_id. Does NOT start processing yet."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        return jsonify({"error": f"Unsupported format '{ext}'"}), 400

    job_id = str(uuid.uuid4())[:8]
    saved  = UPLOAD_FOLDER / f"{job_id}_input{ext}"
    f.save(str(saved))

    file_type = "video" if ext in VIDEO_EXTS else "image"
    with jobs_lock:
        jobs[job_id] = {
            "status":        "waiting",
            "progress":      0,
            "stats":         {},
            "total":         0,
            "aoi_stats":     {},
            "has_aoi":       False,
            "file_type":     file_type,
            "original_name": f.filename,
            "input_file":    str(saved),
        }

    return jsonify({"job_id": job_id, "file_type": file_type})


@app.route("/thumbnail/<job_id>")
def thumbnail(job_id):
    """Return first video frame (or image) as JPEG for AOI canvas drawing."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    filepath  = job.get("input_file", "")
    file_type = job.get("file_type", "image")

    if not Path(filepath).exists():
        return jsonify({"error": "Input file not found"}), 404

    if file_type == "video":
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({"error": "Could not read video frame"}), 500
    else:
        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({"error": "Could not read image"}), 500

    # Resize to max 960px wide for fast delivery
    h, w = frame.shape[:2]
    if w > 960:
        scale = 960 / w
        frame = cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_AREA)

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return Response(jpeg.tobytes(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


@app.route("/start/<job_id>", methods=["POST"])
def start(job_id):
    """Start processing with optional AOI polygon (normalized coords)."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") != "waiting":
        return jsonify({"error": "Job already started or completed"}), 400

    data       = request.get_json(silent=True) or {}
    aoi_points = data.get("aoi")          # [[nx, ny], ...] or null

    filepath  = job.get("input_file", "")
    file_type = job.get("file_type", "image")

    if not Path(filepath).exists():
        return jsonify({"error": "Upload file not found"}), 404

    has_aoi = bool(aoi_points and len(aoi_points) >= 3)
    update_job(job_id, status="processing", has_aoi=has_aoi)

    fn = process_video if file_type == "video" else process_image
    t  = threading.Thread(target=fn, args=(job_id, filepath, aoi_points), daemon=True)
    t.start()

    return jsonify({"ok": True, "has_aoi": has_aoi})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/stream/<job_id>")
def stream(job_id):
    """MJPEG live stream of annotated frames while processing."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return Response(
        generate_mjpeg(job_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/preview/<job_id>")
def preview(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    out_file = OUTPUT_FOLDER / job["output_file"]
    if not out_file.exists():
        return jsonify({"error": "Output file missing"}), 404
    return send_file(str(out_file), conditional=True)


@app.route("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    out_file  = OUTPUT_FOLDER / job["output_file"]
    orig_name = Path(job.get("original_name", "result"))
    dl_name   = f"detected_{orig_name.stem}{orig_name.suffix}"
    return send_file(str(out_file), as_attachment=True, download_name=dl_name)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  🚗  Object Detection & Tracking Dashboard")
    print("=" * 50)
    print(f"  🌐  Open: http://localhost:8080")
    print(f"  📁  Uploads  → {UPLOAD_FOLDER}")
    print(f"  📤  Outputs  → {OUTPUT_FOLDER}")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
