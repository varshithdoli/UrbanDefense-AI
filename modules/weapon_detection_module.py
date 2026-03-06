import cv2
import numpy as np
import os
import time
from collections import OrderedDict

# ══════════════════════════════════════════════════════════
#  SURVEILLANCE-GRADE WEAPON DETECTION MODULE
# ══════════════════════════════════════════════════════════
#
#  Features:
#    1. Multi-Model Pipeline:
#       - YOLOv8s (default COCO) -> For general objects (used by other modules)
#       - YOLOv8s-Pose -> For person detection & hand keypoint extraction
#       - Custom YOLO model -> For weapon detection
#    2. Pose Analysis (Hand-Object Relationship):
#       - Cross-references weapon bounding boxes with person wrists
#       - Ignores weapons not held by a person (e.g., on tables, posters)
#    3. Multi-Frame Confirmation:
#       - Tracks persons across frames using CentroidTracker
#       - Requires weapon to be held by SAME person for N consecutive frames
#
# ══════════════════════════════════════════════════════════

try:
    from ultralytics import YOLO
    
    # Base COCO model (Class 0: Person, 43: Knife, 34: Bat, 76: Scissors)
    # NOTE: Exported as 'model' so other modules (Traffic, Abandoned Object) can import it safely
    # We use YOLOv8s for better CCTV accuracy as requested.
    model = YOLO("yolov8s.pt")
    
    # Pose model for human hand-object relationship
    # YOLOv8s-pose gives 17 keypoints. 9: left wrist, 10: right wrist
    pose_model = YOLO("yolov8s-pose.pt")
    
    # Weapon model (user's custom trained model for firearms)
    CUSTOM_MODEL_PATH = "Models/weapon_model.pt"
    if os.path.exists(CUSTOM_MODEL_PATH):
        weapon_model = YOLO(CUSTOM_MODEL_PATH)
        DANGEROUS_CLASSES = None  # Custom model = detect all its classes
        print(f"[WEAPON] Loaded custom WEAPON model from {CUSTOM_MODEL_PATH}")
    else:
        weapon_model = model  # Fallback to base model
        DANGEROUS_CLASSES = [43, 34, 76]  # Knife, Bat, Scissors
        print("[WEAPON] Using default COCO model for weapons. Please train and add a custom gun model.")

except Exception as e:
    print(f"Failed to load YOLO models: {e}")
    model = None
    pose_model = None
    weapon_model = None
    DANGEROUS_CLASSES = []

# ── Configurable Parameters ──
CONF_THRESHOLD = 0.5            # Minimum confidence for weapon detection
HAND_PROXIMITY_PX = 80          # Pixels: Max distance between weapon center and wrist keypoint
BBOX_OVERLAP_MARGIN = 20        # Pixels: Fallback margin to check if weapon is inside person bbox
FRAMES_TO_CONFIRM = 3           # Must detect weapon on same person for 3 consecutive frames
MAX_DISAPPEAR_FRAMES = 15       # Forget tracking if person disappears for this long

# ══════════════════════════════════════════════════════════
#  SIMPLE CENTROID TRACKER (For Persons)
# ══════════════════════════════════════════════════════════
class PersonTracker:
    def __init__(self):
        self.next_id = 0
        self.persons = OrderedDict()       # id -> centroid (cx, cy)
        self.bboxes = OrderedDict()        # id -> bbox (x1,y1,x2,y2)
        self.threat_streaks = OrderedDict()# id -> consecutive frames holding weapon
        self.disappeared = OrderedDict()   # id -> frames not seen
        
    def update(self, detections):
        """
        detections: list of (centroid, bbox)
        Returns: dict -> id: (centroid, bbox, threat_streak)
        """
        if len(detections) == 0:
            for pid in list(self.disappeared.keys()):
                self.disappeared[pid] += 1
                if self.disappeared[pid] > MAX_DISAPPEAR_FRAMES:
                    self.deregister(pid)
            return {pid: (self.persons[pid], self.bboxes[pid], self.threat_streaks[pid]) for pid in self.persons}

        input_centroids = [d[0] for d in detections]
        input_bboxes = [d[1] for d in detections]

        if len(self.persons) == 0:
            for i in range(len(detections)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            obj_ids = list(self.persons.keys())
            obj_centroids = list(self.persons.values())

            D = np.zeros((len(obj_centroids), len(input_centroids)), dtype=np.float32)
            for i, oc in enumerate(obj_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.sqrt((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 150: # max tracking distance
                    continue

                pid = obj_ids[row]
                self.persons[pid] = input_centroids[col]
                self.bboxes[pid] = input_bboxes[col]
                self.disappeared[pid] = 0

                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(len(obj_centroids))) - used_rows:
                pid = obj_ids[row]
                self.disappeared[pid] += 1
                if self.disappeared[pid] > MAX_DISAPPEAR_FRAMES:
                    self.deregister(pid)

            for col in set(range(len(input_centroids))) - used_cols:
                self.register(input_centroids[col], input_bboxes[col])

        return {pid: (self.persons[pid], self.bboxes[pid], self.threat_streaks[pid]) for pid in self.persons}

    def register(self, centroid, bbox):
        self.persons[self.next_id] = centroid
        self.bboxes[self.next_id] = bbox
        self.threat_streaks[self.next_id] = 0
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, pid):
        del self.persons[pid]
        del self.bboxes[pid]
        del self.threat_streaks[pid]
        del self.disappeared[pid]

    def increment_threat(self, pid):
        if pid in self.threat_streaks:
            self.threat_streaks[pid] += 1
            return self.threat_streaks[pid]
        return 0
        
    def reset_threat(self, pid):
        if pid in self.threat_streaks:
            if self.threat_streaks[pid] > 0:
                self.threat_streaks[pid] -= 1  # decay rather than hard reset to allow 1 frame misses

# ══════════════════════════════════════════════════════════
#  MODULE STATE
# ══════════════════════════════════════════════════════════
tracker = PersonTracker()

def reset_state():
    global tracker
    tracker = PersonTracker()

def _distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# ══════════════════════════════════════════════════════════
#  MAIN PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════
def process_frame_for_weapons(frame, conf_threshold=CONF_THRESHOLD):
    threats = []
    if model is None or pose_model is None or weapon_model is None:
        return frame, threats

    # ── 1. Detect Weapons ──
    # Run the dedicated weapon model
    w_results = weapon_model(frame, verbose=False, conf=conf_threshold)
    detected_weapons = [] # [(cls_name, conf, x1,y1,x2,y2, cx,cy), ...]
    
    for result in w_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if DANGEROUS_CLASSES is None or cls in DANGEROUS_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                c_name = weapon_model.names[cls]
                c_conf = float(box.conf[0])
                detected_weapons.append((c_name, c_conf, x1, y1, x2, y2, cx, cy))

    # If no weapons at all, decay all active threat streaks and return
    if not detected_weapons:
        for pid in list(tracker.persons.keys()):
            tracker.reset_threat(pid)
        return frame, threats

    # ── 2. Detect Persons & Pose (Hand Keypoints) ──
    p_results = pose_model(frame, verbose=False, conf=0.4)
    person_detections = [] # [(centroid, bbox, wrists)]
    
    for result in p_results:
        if result.keypoints is None: continue
        
        boxes = result.boxes
        keypoints_tensor = result.keypoints.xy # (N, 17, 2)
        
        for i, box in enumerate(boxes):
            if int(box.cls[0]) == 0: # Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Extract wrists (index 9 = left wrist, index 10 = right wrist)
                wrists = []
                if len(keypoints_tensor) > i:
                    kps = keypoints_tensor[i]
                    if len(kps) > 10:
                        lw = (int(kps[9][0]), int(kps[9][1]))
                        rw = (int(kps[10][0]), int(kps[10][1]))
                        if lw != (0,0): wrists.append(lw)
                        if rw != (0,0): wrists.append(rw)
                
                person_detections.append(((cx, cy), (x1, y1, x2, y2), wrists))

    # ── 3. Update Person Tracker ──
    tracked_persons = tracker.update([(d[0], d[1]) for d in person_detections])
    
    # Map back wrists to tracked persons based on bounding box match
    person_wrists = {}
    for (cx, cy), bbox, wrists in person_detections:
        for pid, (t_cx, t_cy, t_score) in tracker.persons.items():
            t_bbox = tracker.bboxes[pid]
            if bbox == t_bbox: # Exact match since we just updated
                person_wrists[pid] = wrists
                break

    # ── 4. Spatial Relationship Analysis ──
    # Check if any weapon is being held by a tracked person
    armed_persons = set()
    
    for w_name, w_conf, wx1, wy1, wx2, wy2, wcx, wcy in detected_weapons:
        weapon_held = False
        holding_pid = None
        
        for pid, (p_cx, p_cy, streak) in tracked_persons.items():
            px1, py1, px2, py2 = tracker.bboxes[pid]
            wrists = person_wrists.get(pid, [])
            
            # Check 1: Is weapon near wrist?
            near_wrist = False
            for w in wrists:
                if _distance((wcx, wcy), w) < HAND_PROXIMITY_PX:
                    near_wrist = True
                    break
                    
            # Check 2: Is weapon heavily inside person's bounding box? (Fallback if pose fails)
            inside_bbox = (wx1 >= px1 - BBOX_OVERLAP_MARGIN and wx2 <= px2 + BBOX_OVERLAP_MARGIN and 
                           wy1 >= py1 - BBOX_OVERLAP_MARGIN and wy2 <= py2 + BBOX_OVERLAP_MARGIN)
            
            if near_wrist or inside_bbox:
                weapon_held = True
                holding_pid = pid
                break # Weapon belongs to this person

        if weapon_held and holding_pid is not None:
            armed_persons.add((holding_pid, w_name, w_conf, wx1, wy1, wx2, wy2))
            
            # ── Draw WEAPON Context ──
            # Draw cyan context line from person centroid to weapon
            pcx, pcy = tracker.persons[holding_pid]
            cv2.line(frame, (pcx, pcy), (wcx, wcy), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 165, 255), 2)
            cv2.putText(frame, f"{w_name} {w_conf:.2f}", (wx1, wy1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            # Weapon NOT held by anyone (e.g., on table) -> Ignore or mark safely
            cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (180, 180, 180), 1)
            cv2.putText(frame, f"Unattended {w_name}", (wx1, wy1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ── 5. Multi-Frame Confirmation & Alerts ──
    for pid in list(tracker.persons.keys()):
        # Find if this person is currently armed
        armed_data = None
        for ap in armed_persons:
            if ap[0] == pid:
                armed_data = ap
                break
                
        px1, py1, px2, py2 = tracker.bboxes[pid]
        
        if armed_data:
            pid, w_name, w_conf, wx1, wy1, wx2, wy2 = armed_data
            streak = tracker.increment_threat(pid)
            
            if streak >= FRAMES_TO_CONFIRM:
                # 🚨 SURVEILLANCE-GRADE ARMED THREAT 🚨
                threat_msg = f"Confirmed Armed Threat: Person carrying {w_name} (Conf: {w_conf:.2f})"
                if threat_msg not in threats:
                    threats.append(threat_msg)
                
                # Draw RED pulsing target box on the person
                pulse = int(50 * np.sin(time.time() * 10) + 50)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                
                # Draw Threat Label
                label = f"ARMED THREAT [{w_name.upper()}]"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (px1, py1 - label_size[1] - 10), (px1 + label_size[0] + 10, py1), (0, 0, 255), -1)
                cv2.putText(frame, label, (px1 + 5, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Orange Warning (building up frames)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)
                cv2.putText(frame, f"Tracking Weapon... ({streak}/{FRAMES_TO_CONFIRM})", (px1, py1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            # Person not armed this frame, decay threat
            tracker.reset_threat(pid)
            streak = tracker.threat_streaks[pid]
            
            # Green Normal Box
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 1)
            if streak > 0:
                cv2.putText(frame, f"Weapon Lost ({streak}/{FRAMES_TO_CONFIRM})", (px1, py1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return frame, threats
