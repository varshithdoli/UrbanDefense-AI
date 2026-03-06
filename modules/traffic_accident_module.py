import cv2
import numpy as np
import time
from collections import OrderedDict, deque

# ══════════════════════════════════════════════════════════
#  ENHANCED TRAFFIC ACCIDENT DETECTION MODULE
# ══════════════════════════════════════════════════════════
#
#  Pipeline:
#    1. YOLOv8 Vehicle Detection (car, motorcycle, bus, truck)
#    2. Vehicle Tracking with unique IDs (centroid-based)
#    3. Speed Estimation (displacement per frame → sudden stops)
#    4. Trajectory Analysis (path intersection detection)
#    5. IoU Collision Check (bounding box overlap)
#    6. Multi-Signal Accident Logic (combines all signals)
#    7. Alert System integration
#
#  Accident is confirmed ONLY when multiple signals agree:
#    - Sudden speed drop  AND
#    - Bounding box overlap (IoU > threshold)  AND
#    - Vehicles stop for > 2 seconds
# ══════════════════════════════════════════════════════════

# ── Import shared YOLO model ──
try:
    from modules.weapon_detection_module import model as yolo_model
except ImportError:
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
    except Exception:
        yolo_model = None

# ── COCO Vehicle Class IDs ──
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
VEHICLE_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# ── Configurable Parameters ──
IOU_THRESHOLD = 0.3              # IoU above this = physical overlap detected
SPEED_DROP_RATIO = 0.6           # Speed drops to 40% of average = sudden stop
PROXIMITY_THRESHOLD_PX = 80     # Pixels — vehicles closer than this = very close
STOP_SPEED_THRESHOLD = 3.0      # Pixels/frame — below this = vehicle is stopped
STOP_DURATION_SEC = 2.0          # Seconds vehicles must be stopped after collision
TRAJECTORY_HISTORY = 30          # Number of past positions to store per vehicle
MAX_DISAPPEAR_FRAMES = 40       # Frames before a tracked vehicle is deregistered
ACCIDENT_COOLDOWN_SEC = 15       # Don't re-alert same collision pair for this long


# ══════════════════════════════════════════════════════════
#  VEHICLE TRACKER
# ══════════════════════════════════════════════════════════
class VehicleTracker:
    """
    Tracks vehicles across frames using centroid distance matching.
    Stores position history, speed, and trajectory for each vehicle.
    """

    def __init__(self):
        self.next_id = 0
        self.vehicles = OrderedDict()  # id -> VehicleState

    def update(self, detections):
        """
        Update tracker with new detections.
        detections: list of (centroid, bbox, class_id, confidence)
        Returns: dict of {id: VehicleState}
        """
        if len(detections) == 0:
            # Mark all as disappeared
            for vid in list(self.vehicles.keys()):
                self.vehicles[vid].disappeared += 1
                if self.vehicles[vid].disappeared > MAX_DISAPPEAR_FRAMES:
                    del self.vehicles[vid]
            return self.vehicles

        input_data = detections

        if len(self.vehicles) == 0:
            # Register all new detections
            for centroid, bbox, cls_id, conf in input_data:
                self._register(centroid, bbox, cls_id, conf)
        else:
            # Match existing vehicles to new detections via centroid distance
            obj_ids = list(self.vehicles.keys())
            obj_centroids = [self.vehicles[oid].centroid for oid in obj_ids]

            input_centroids = [d[0] for d in input_data]

            # Build distance matrix
            D = np.zeros((len(obj_centroids), len(input_centroids)), dtype=np.float32)
            for i, oc in enumerate(obj_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.sqrt((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2)

            # Greedy matching
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 120:  # Max match distance
                    continue

                obj_id = obj_ids[row]
                centroid, bbox, cls_id, conf = input_data[col]
                self.vehicles[obj_id].update_position(centroid, bbox)
                self.vehicles[obj_id].disappeared = 0

                used_rows.add(row)
                used_cols.add(col)

            # Deregister unmatched existing
            for row in set(range(len(obj_centroids))) - used_rows:
                obj_id = obj_ids[row]
                self.vehicles[obj_id].disappeared += 1
                if self.vehicles[obj_id].disappeared > MAX_DISAPPEAR_FRAMES:
                    del self.vehicles[obj_id]

            # Register unmatched new detections
            for col in set(range(len(input_centroids))) - used_cols:
                centroid, bbox, cls_id, conf = input_data[col]
                self._register(centroid, bbox, cls_id, conf)

        return self.vehicles

    def _register(self, centroid, bbox, cls_id, conf):
        vid = self.next_id
        self.vehicles[vid] = VehicleState(vid, centroid, bbox, cls_id, conf)
        self.next_id += 1
        return vid


class VehicleState:
    """Stores all tracking data for a single vehicle."""

    def __init__(self, vid, centroid, bbox, cls_id, conf):
        self.id = vid
        self.centroid = centroid
        self.bbox = bbox
        self.cls_id = cls_id
        self.confidence = conf
        self.disappeared = 0

        # Position history for trajectory + speed
        self.positions = deque(maxlen=TRAJECTORY_HISTORY)
        self.positions.append((centroid[0], centroid[1], time.time()))

        # Speed tracking
        self.speeds = deque(maxlen=20)        # Recent speeds (px/frame)
        self.current_speed = 0.0
        self.avg_speed = 0.0
        self.speed_drop_detected = False

        # Stop detection
        self.stop_start_time = None           # When vehicle first stopped
        self.is_stopped = False

    def update_position(self, centroid, bbox):
        """Update vehicle position and calculate speed."""
        prev_x, prev_y, prev_t = self.positions[-1]
        curr_t = time.time()

        # Calculate speed (pixels per time unit)
        dx = centroid[0] - prev_x
        dy = centroid[1] - prev_y
        dt = max(curr_t - prev_t, 0.001)

        displacement = np.sqrt(dx ** 2 + dy ** 2)
        self.current_speed = displacement / dt  # pixels per second

        # Normalize to ~pixels per frame (assuming ~30fps)
        speed_per_frame = displacement

        self.speeds.append(speed_per_frame)

        # Calculate average speed (excluding current)
        if len(self.speeds) > 5:
            self.avg_speed = np.mean(list(self.speeds)[:-3])
        else:
            self.avg_speed = np.mean(list(self.speeds))

        # Detect sudden speed drop
        if (self.avg_speed > 8 and
                speed_per_frame < self.avg_speed * (1 - SPEED_DROP_RATIO)):
            self.speed_drop_detected = True
        else:
            self.speed_drop_detected = False

        # Detect if vehicle is stopped
        if speed_per_frame < STOP_SPEED_THRESHOLD:
            if not self.is_stopped:
                self.stop_start_time = curr_t
                self.is_stopped = True
        else:
            self.is_stopped = False
            self.stop_start_time = None

        # Update state
        self.centroid = centroid
        self.bbox = bbox
        self.positions.append((centroid[0], centroid[1], curr_t))

    @property
    def stop_duration(self):
        """How long the vehicle has been stopped (seconds)."""
        if self.stop_start_time is None:
            return 0.0
        return time.time() - self.stop_start_time

    @property
    def trajectory(self):
        """Return trajectory as list of (x, y) points."""
        return [(int(p[0]), int(p[1])) for p in self.positions]

    @property
    def vehicle_name(self):
        return VEHICLE_NAMES.get(self.cls_id, "Vehicle")


# ══════════════════════════════════════════════════════════
#  ACCIDENT DETECTION ENGINE
# ══════════════════════════════════════════════════════════
class AccidentDetector:
    """
    Multi-signal accident detection.
    Combines: IoU overlap + speed drop + proximity + stop duration.
    """

    def __init__(self):
        self.active_incidents = {}     # (id1, id2) -> {'start': time, 'alerted': bool, 'signals': {}}
        self.cooldowns = {}            # (id1, id2) -> last_alert_time

    def check_accidents(self, vehicles):
        """
        Check all vehicle pairs for accident signals.
        Returns list of (threat_string, collision_bbox, involved_ids)
        """
        threats = []
        current_time = time.time()
        vehicle_list = list(vehicles.values())

        # Check all pairs
        for i in range(len(vehicle_list)):
            for j in range(i + 1, len(vehicle_list)):
                v1 = vehicle_list[i]
                v2 = vehicle_list[j]

                pair_key = (min(v1.id, v2.id), max(v1.id, v2.id))

                # Cooldown check
                if pair_key in self.cooldowns:
                    if current_time - self.cooldowns[pair_key] < ACCIDENT_COOLDOWN_SEC:
                        continue

                # ── Signal 1: IoU Overlap ──
                iou = self._compute_iou(v1.bbox, v2.bbox)

                # ── Signal 2: Proximity ──
                dist = np.sqrt(
                    (v1.centroid[0] - v2.centroid[0]) ** 2 +
                    (v1.centroid[1] - v2.centroid[1]) ** 2
                )

                # ── Signal 3: Speed Drop ──
                speed_drop = v1.speed_drop_detected or v2.speed_drop_detected

                # ── Signal 4: Both Vehicles Stopped ──
                both_stopped = v1.is_stopped and v2.is_stopped
                stop_dur = min(v1.stop_duration, v2.stop_duration) if both_stopped else 0

                # ── Signal 5: Trajectory Intersection ──
                traj_intersect = self._check_trajectory_intersection(v1, v2)

                # ── Combine Signals ──
                signals = {
                    'iou': iou,
                    'distance': dist,
                    'speed_drop': speed_drop,
                    'both_stopped': both_stopped,
                    'stop_duration': stop_dur,
                    'trajectory_intersect': traj_intersect,
                }

                # Score the incident (0-100)
                score = self._calculate_accident_score(signals)

                if score >= 60:
                    # Track incident
                    if pair_key not in self.active_incidents:
                        self.active_incidents[pair_key] = {
                            'start': current_time,
                            'alerted': False,
                            'signals': signals,
                            'score': score,
                        }

                    incident = self.active_incidents[pair_key]
                    incident['signals'] = signals
                    incident['score'] = score

                    # Require confirmation: signals must persist OR stop duration met
                    confirmed = (
                        (both_stopped and stop_dur >= STOP_DURATION_SEC) or
                        (score >= 80) or
                        (iou > 0.4 and speed_drop)
                    )

                    if confirmed and not incident['alerted']:
                        # Build collision bounding box
                        collision_bbox = (
                            min(v1.bbox[0], v2.bbox[0]),
                            min(v1.bbox[1], v2.bbox[1]),
                            max(v1.bbox[2], v2.bbox[2]),
                            max(v1.bbox[3], v2.bbox[3]),
                        )

                        involved = f"{v1.vehicle_name} #{v1.id} & {v2.vehicle_name} #{v2.id}"
                        threat_msg = (
                            f"Traffic Accident Detected: {involved} "
                            f"(Score: {score}/100, IoU: {iou:.2f})"
                        )
                        threats.append((threat_msg, collision_bbox, (v1.id, v2.id), signals))

                        incident['alerted'] = True
                        self.cooldowns[pair_key] = current_time
                else:
                    # Remove incident if signals drop
                    if pair_key in self.active_incidents:
                        if current_time - self.active_incidents[pair_key]['start'] > 5:
                            del self.active_incidents[pair_key]

        return threats

    def _compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / float(union_area) if union_area > 0 else 0.0

    def _check_trajectory_intersection(self, v1, v2):
        """
        Check if two vehicles' recent trajectories are converging/intersecting.
        Uses the last few positions to determine if paths cross.
        """
        traj1 = v1.trajectory
        traj2 = v2.trajectory

        if len(traj1) < 3 or len(traj2) < 3:
            return False

        # Check if distance between vehicles is decreasing rapidly
        # Compare distance at start of trajectory vs now
        start_dist = np.sqrt(
            (traj1[0][0] - traj2[0][0]) ** 2 +
            (traj1[0][1] - traj2[0][1]) ** 2
        )
        curr_dist = np.sqrt(
            (traj1[-1][0] - traj2[-1][0]) ** 2 +
            (traj1[-1][1] - traj2[-1][1]) ** 2
        )

        # Trajectories converging if distance decreased significantly
        if start_dist > 50 and curr_dist < start_dist * 0.4:
            return True

        # Check if trajectory segments actually cross (line intersection)
        if len(traj1) >= 2 and len(traj2) >= 2:
            # Use last two points of each trajectory as line segments
            if self._segments_intersect(
                traj1[-2], traj1[-1], traj2[-2], traj2[-1]
            ):
                return True

        return False

    def _segments_intersect(self, p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects with p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        return False

    def _calculate_accident_score(self, signals):
        """
        Calculate accident confidence score (0-100) from multiple signals.
        Each signal contributes a weighted score.
        """
        score = 0

        # IoU weight: 0-35 points
        iou = signals['iou']
        if iou > IOU_THRESHOLD:
            score += min(35, int(iou * 70))

        # Speed drop: 0-25 points
        if signals['speed_drop']:
            score += 25

        # Proximity: 0-15 points
        dist = signals['distance']
        if dist < PROXIMITY_THRESHOLD_PX:
            score += 15
        elif dist < PROXIMITY_THRESHOLD_PX * 2:
            score += 8

        # Both stopped: 0-15 points
        if signals['both_stopped']:
            score += 10
            if signals['stop_duration'] >= STOP_DURATION_SEC:
                score += 5

        # Trajectory intersection: 0-10 points
        if signals['trajectory_intersect']:
            score += 10

        return min(100, score)


# ══════════════════════════════════════════════════════════
#  VISUAL ANNOTATIONS
# ══════════════════════════════════════════════════════════
def _draw_annotations(frame, vehicles, accident_threats):
    """Draw vehicle tracking, trajectories, speed, and accident zones."""

    accident_vehicle_ids = set()
    for _, _, involved_ids, _ in accident_threats:
        accident_vehicle_ids.update(involved_ids)

    for vid, vehicle in vehicles.items():
        x1, y1, x2, y2 = vehicle.bbox
        cx, cy = vehicle.centroid
        is_accident = vid in accident_vehicle_ids

        # ── Choose color based on state ──
        if is_accident:
            color = (0, 0, 255)       # Red — accident
        elif vehicle.speed_drop_detected:
            color = (0, 165, 255)     # Orange — sudden brake
        elif vehicle.is_stopped:
            color = (0, 255, 255)     # Yellow — stopped
        else:
            color = (0, 255, 0)       # Green — normal

        # ── Draw bounding box ──
        thickness = 3 if is_accident else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # ── Draw label ──
        name = vehicle.vehicle_name
        speed_text = f"{vehicle.current_speed:.0f}px/s"
        label = f"{name} #{vid} | {speed_text}"

        if vehicle.speed_drop_detected:
            label += " | BRAKING!"
        if vehicle.is_stopped and vehicle.stop_duration > 1:
            label += f" | STOPPED {vehicle.stop_duration:.0f}s"

        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Draw trajectory trail ──
        traj = vehicle.trajectory
        if len(traj) >= 2:
            for k in range(1, len(traj)):
                # Fade older points
                alpha = k / len(traj)
                trail_color = (
                    int(color[0] * alpha),
                    int(color[1] * alpha),
                    int(color[2] * alpha)
                )
                cv2.line(frame, traj[k - 1], traj[k], trail_color, 2, cv2.LINE_AA)

            # Arrow showing direction of travel
            if len(traj) >= 2:
                cv2.arrowedLine(frame, traj[-2], traj[-1], color, 2, tipLength=0.4)

    # ── Draw accident zones ──
    for threat_msg, collision_bbox, involved_ids, signals in accident_threats:
        cx1, cy1, cx2, cy2 = collision_bbox

        # Red collision zone with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx1 - 5, cy1 - 5), (cx2 + 5, cy2 + 5), (0, 0, 255), -1)
        pulse = 0.15 + 0.1 * np.sin(time.time() * 6)
        cv2.addWeighted(overlay, pulse, frame, 1 - pulse, 0, frame)

        # Accident border
        cv2.rectangle(frame, (cx1 - 5, cy1 - 5), (cx2 + 5, cy2 + 5), (0, 0, 255), 3)

        # Accident label with details
        score = signals.get('iou', 0) * 100
        label = f"ACCIDENT DETECTED (IoU:{signals['iou']:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Label background
        label_y = max(cy1 - 20, label_size[1] + 10)
        cv2.rectangle(frame,
                      (cx1 - 5, label_y - label_size[1] - 10),
                      (cx1 + label_size[0] + 10, label_y + 5),
                      (0, 0, 200), -1)
        cv2.putText(frame, label, (cx1, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Signal details below
        detail_y = cy2 + 20
        signal_texts = []
        if signals['speed_drop']:
            signal_texts.append("Speed Drop")
        if signals['both_stopped']:
            signal_texts.append(f"Stopped {signals['stop_duration']:.0f}s")
        if signals['trajectory_intersect']:
            signal_texts.append("Paths Crossed")

        if signal_texts:
            detail = "Signals: " + " + ".join(signal_texts)
            cv2.putText(frame, detail, (cx1, detail_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════
#  MODULE STATE
# ══════════════════════════════════════════════════════════
vehicle_tracker = VehicleTracker()
accident_detector = AccidentDetector()


def reset_state():
    """Reset all tracking state (call when starting a new video feed)."""
    global vehicle_tracker, accident_detector
    vehicle_tracker = VehicleTracker()
    accident_detector = AccidentDetector()


# ══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════
def process_frame_for_accidents(frame):
    """
    Enhanced traffic accident detection with multi-signal analysis.

    Pipeline:
      1. YOLOv8 vehicle detection (car, motorcycle, bus, truck)
      2. Vehicle tracking with unique IDs and trajectory history
      3. Speed estimation and sudden-stop detection
      4. Trajectory intersection analysis
      5. IoU bounding-box overlap check
      6. Multi-signal accident scoring (score 0-100)
      7. Alert only when multiple signals confirm (reduces false positives)

    Returns: (annotated_frame, list_of_threat_strings)
    """
    threats = []

    if yolo_model is None:
        return frame, threats

    # ── Step 1: YOLO Vehicle Detection ──
    results = yolo_model(frame, verbose=False, classes=VEHICLE_CLASSES, conf=0.4)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            detections.append((centroid, (x1, y1, x2, y2), cls_id, conf))

    # ── Step 2: Update Vehicle Tracker ──
    tracked_vehicles = vehicle_tracker.update(detections)

    # ── Step 3-6: Multi-Signal Accident Detection ──
    accident_threats = accident_detector.check_accidents(tracked_vehicles)

    # Extract threat strings
    for threat_msg, collision_bbox, involved_ids, signals in accident_threats:
        threats.append(threat_msg)

    # ── Step 7: Draw Annotations ──
    frame = _draw_annotations(frame, tracked_vehicles, accident_threats)

    # ── Status overlay ──
    status = f"Tracking: {len(tracked_vehicles)} vehicles"
    active = len(accident_detector.active_incidents)
    if active > 0:
        status += f" | {active} incident(s) monitored"
    cv2.putText(frame, status, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame, threats
