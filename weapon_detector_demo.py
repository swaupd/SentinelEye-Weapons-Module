# weapon_detector_demo.py
import argparse
import time
import json
from collections import deque
from datetime import datetime
import os

import cv2
import numpy as np
from imutils.video import FPS
import torch
from ultralytics import YOLO  # pip install ultralytics

# ---------- CONFIG ----------
BUFFER_SECONDS = 6               # how many seconds to buffer before/after trigger
FPS_TARGET = 20                  # assumed fps for buffer sizing
SAVE_CLIP_SECONDS_AFTER = 4      # seconds to continue recording after trigger
MIN_CONFIDENCE = 0.35            # min detection confidence to consider
IOU_PERSON_WEAPON = 0.03         # overlap threshold for associating weapon with person (low to be permissive)

OUTPUT_DIR = "evidence"
LOG_FILE = os.path.join(OUTPUT_DIR, "threats.log")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map model class names -> our canonical labels and threat levels.
# Edit this map if your model uses different class names.
CLASS_MAP = {
    "handgun": ("Gun", "critical"),
    "Pistol": ("Gun", "critical"),
    "knife": ("Knife", "high"),
    "bat": ("Bat", "high"),
    "mask": ("MaskedPerson", "suspicious"),
    "backpack": ("Backpack", "monitor"),  # e.g., unattended bag -> monitor/medium
    # add more mappings if your model has more classes
}

# Person label the model must emit. If different, change here.
PERSON_CLASS_NAME = "person"
# ----------------------------

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def save_frame_image(frame, prefix="evidence"):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
    path = os.path.join(OUTPUT_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path

def save_buffer_as_video(frames, out_path, fps=FPS_TARGET):
    if len(frames) == 0:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return True

def bbox_iou_area_overlap(boxA, boxB):
    # boxes are (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    # use ratio to person area (weapon touching person)
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / areaA, interArea / areaB

def log_threat(entry):
    # JSONL append
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 weights or model (local path or model name)")
    p.add_argument("--source", type=str, default="0", help="Video source (0 for webcam or path to video)")
    p.add_argument("--device", type=str, default="cpu", help="Computation device for ultralytics (0 or 'cpu')")
    p.add_argument("--buffer_seconds", type=int, default=BUFFER_SECONDS)
    p.add_argument("--fps", type=int, default=FPS_TARGET)
    return p.parse_args()

def main():
    args = parse_args()
    # load model
    print(f"[+] Loading model {args.model} on device {args.device} ...")
    model = YOLO(args.model)
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)

    # open video source
    source = int(args.source) if args.source.isnumeric() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: cannot open video source", source)
        return

    # Try to determine FPS
    fps = args.fps
    try:
        file_fps = cap.get(cv2.CAP_PROP_FPS)
        if file_fps and file_fps > 1:
            fps = int(file_fps)
    except Exception:
        pass
    buffer_size = max(1, int(args.buffer_seconds * fps))
    frame_buffer = deque(maxlen=buffer_size)
    post_trigger_remaining = 0

    fps_meter = FPS().start()
    print("[+] Starting detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_frame = frame.copy()
        frame_buffer.append(orig_frame)
        # run detection (resize to make it faster for demo)
        # ultralytics model automatically resizes input
        results = model(orig_frame, conf=MIN_CONFIDENCE, device=device, verbose=False)
        # results is a list; take first result object
        r = results[0]
        boxes = r.boxes  # Boxes object
        # Parse detections into list: (label_name, conf, bbox)
        detections = []
        for b in boxes:
            cls_id = int(b.cls)
            conf = float(b.conf)
            xyxy = list(map(int, b.xyxy[0]))  # [x1,y1,x2,y2]
            # get label name (model.names)
            label_name = r.names.get(cls_id, str(cls_id))
            detections.append((label_name, conf, xyxy))

        # separate persons and others
        persons = []
        others = []
        for label, conf, xyxy in detections:
            if label.lower() == PERSON_CLASS_NAME.lower():
                persons.append((label, conf, xyxy))
            else:
                others.append((label, conf, xyxy))

        # threat detection logic: look for weapon classes in others that are mapped in CLASS_MAP
        threats_found = []
        for label, conf, xyxy in others:
            ll = label.lower()
            if ll in CLASS_MAP and conf >= MIN_CONFIDENCE:
                canon_label, default_level = CLASS_MAP[ll]
                # check association to person: if overlap with any person bbox
                assoc_person = None
                assoc_score = 0.0
                for p_label, p_conf, p_xyxy in persons:
                    overlap_p_weapon, overlap_weapon_p = bbox_iou_area_overlap(p_xyxy, xyxy)
                    # use small thresholds: if some overlap with person area, weapon is likely in-hand / near person
                    if overlap_p_weapon > IOU_PERSON_WEAPON or overlap_weapon_p > IOU_PERSON_WEAPON:
                        if overlap_weapon_p > assoc_score:
                            assoc_score = overlap_weapon_p
                            assoc_person = p_xyxy
                # decide threat level: if associated_with_person => escalate
                level = default_level
                if assoc_person is not None:
                    # escalate suspicious->high/critical for weapons when near a person
                    if level == "suspicious":
                        level = "high"
                    elif level == "monitor":
                        level = "medium"
                    # for guns/knives, default_level likely already high/critical
                # create detection entry
                threats_found.append({
                    "class": canon_label,
                    "raw_label": label,
                    "confidence": float(conf),
                    "bbox": xyxy,
                    "associated_person_overlap": float(assoc_score),
                    "level": level
                })

        triggered = False
        if threats_found:
            # determine highest severity among threats to decide action
            severity_order = {"monitor": 0, "suspicious": 1, "medium": 2, "high": 3, "critical": 4}
            highest = max(threats_found, key=lambda t: severity_order.get(t["level"], 0))
            highest_level = highest["level"]
            # For demo: trigger for medium or higher
            if severity_order.get(highest_level, 0) >= severity_order.get("medium", 2):
                triggered = True

        # If triggered, dump buffer to evidence and set post-trigger counter
        if triggered:
            print(f"[!] Threat detected: saving evidence (level {highest_level})")
            # save last buffer frames as video
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            video_path = os.path.join(OUTPUT_DIR, f"evidence_{ts}.mp4")
            # gather frames: buffer + will include post frames
            # create a local list copy of current buffer; then continue to collect post frames
            collected_frames = list(frame_buffer)
            post_trigger_remaining = int(SAVE_CLIP_SECONDS_AFTER * fps)
            # save snapshot evidence image with boxes drawn (draw now)
            demo_frame = orig_frame.copy()
            for det in detections:
                lab, conf, xyxy = det
                x1, y1, x2, y2 = xyxy
                color = (0, 255, 0)
                # red for mapped threatening classes
                if lab.lower() in CLASS_MAP:
                    color = (0, 0, 255)
                cv2.rectangle(demo_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(demo_frame, f"{lab} {conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            evidence_img_path = save_frame_image(demo_frame, prefix="evidence")
            # log event
            entry = {
                "timestamp": now_iso(),
                "threats": threats_found,
                "level": highest_level,
                "evidence_image": evidence_img_path,
                "evidence_video": video_path,
            }
            log_threat(entry)
            # Start collecting post-trigger frames and then save video once post_trigger_remaining reaches 0
            # We'll reuse collected_frames and append frames while post_trigger_remaining > 0 in the main loop

            # temporarily store this state in a variable so we can access it outside
            buffer_for_saving = collected_frames
            # attach to outer-scope by using a attribute on the cap, hacky but avoids rearchitecting
            saving_buffer = buffer_for_saving
            video_path_for_saving = video_path
            evidence_img_path_var = evidence_img_path

        # If post-trigger is running, decrement and append frames to buffer_for_saving
        if hasattr(cap, "buffer_for_saving") and post_trigger_remaining > 0:
            buffer_for_saving.append(orig_frame)
            post_trigger_remaining -= 1
            # when remaining reaches 0 -> flush to file
            if post_trigger_remaining <= 0:
                # save video
                success = save_buffer_as_video(buffer_for_saving, video_path_for_saving, fps=fps)
                if success:
                    print(f"[+] Saved clip -> {video_path_for_saving}")
                else:
                    print("[!] Failed saving clip")
                # cleanup attributes
                del buffer_for_saving
                del video_path_for_saving
                if hasattr(cap, "evidence_img_path"):
                    del evidence_img_path_var

        # Draw boxes on live frame for demo
        vis = orig_frame.copy()
        for det in detections:
            label, conf, xyxy = det
            x1, y1, x2, y2 = xyxy
            color = (0, 255, 0)
            if label.lower() in CLASS_MAP:
                # color code based on threat
                lvl = CLASS_MAP[label.lower()][1]
                if lvl in ("high", "critical"):
                    color = (0, 0, 255)  # red
                elif lvl in ("medium",):
                    color = (0, 165, 255)  # orange-ish
                else:
                    color = (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{label} {conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # small HUD
        elapsed = time.time() - fps_meter._start.timestamp()
        hud = f"FPS: {fps_meter._numFrames/elapsed if elapsed>0 else 0:.1f}"
        cv2.putText(vis, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Threat Detector", vis)
        key = cv2.waitKey(1) & 0xFF
        fps_meter.update()

        if key == ord("q"):
            break

    fps_meter.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[+] Exited cleanly.")

if __name__ == "__main__":
    main()
