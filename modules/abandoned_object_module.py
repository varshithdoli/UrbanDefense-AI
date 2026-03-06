import cv2
import numpy as np
import time
from collections import OrderedDict

# ══════════════════════════════════════════════════════════
#  ENHANCED ABANDONED OBJECT DETECTION MODULE
# ══════════════════════════════════════════════════════════
#
#  Features:
#    1. Person + Bag detection using YOLO (reuses weapon module's model)
#    2. Centroid-based tracking with unique IDs
#    3. Person-Bag association (nearest person owns the bag)
#    4. Abandonment detection: person leaves bag beyond distance threshold
#    5. Behavioral analysis: leave_object, run_away detection
#
#  COCO classes used:
#    person=0, backpack=24, handbag=26, suitcase=28
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

# ── COCO Class IDs ──
PERSON_CLASS = 0
BAG_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase
ALL_DETECT_CLASSES = [PERSON_CLASS] + BAG_CLASSES

# ── Configurable Parameters ──
ABANDON_DISTANCE_PX = 200       # Pixels — if owner is farther than this, bag is "left behind"
ABANDON_TIME_SEC = 5            # Seconds bag must be unattended before triggering alert
RUN_SPEED_THRESHOLD = 15        # Pixels/frame — speed above this = "running away"
MAX_DISAPPEAR_FRAMES = 50       # Frames before a tracked object is deregistered
ASSOCIATION_DISTANCE_PX = 150   # Max distance to associate a bag with a person initially
LOITER_TIME_SEC = 10            # Seconds of low movement before flagged as loitering
LOITER_RADIUS_PX = 40           # Person must stay within this radius to count as loitering


# ══════════════════════════════════════════════════════════
#  CENTROID TRACKER
# ══════════════════════════════════════════════════════════
class CentroidTracker:
    """
    Assigns and maintains unique IDs for detected objects across frames
    using centroid distance matching.
    """

    def __init__(self, max_disappeared=MAX_DISAPPEAR_FRAMES):
        self.next_id = 0
        self.objects = OrderedDict()        # id -> centroid (cx, cy)
        self.bboxes = OrderedDict()         # id -> (x1, y1, x2, y2)
        self.disappeared = OrderedDict()    # id -> frames since last seen
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.next_id += 1
        return obj_id

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.bboxes[obj_id]
        del self.disappeared[obj_id]

    def update(self, detections):
        """
        Update tracker with new detections.
        detections: list of (centroid, bbox) tuples
        Returns: dict of {id: (centroid, bbox)}
        """
        # If no detections, mark all existing as disappeared
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return {oid: (self.objects[oid], self.bboxes[oid]) for oid in self.objects}

        input_centroids = [d[0] for d in detections]
        input_bboxes = [d[1] for d in detections]

        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            # Compute distance matrix between existing and new centroids
            D = np.zeros((len(obj_centroids), len(input_centroids)), dtype=np.float32)
            for i, oc in enumerate(obj_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.sqrt((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2)

            # Match using greedy closest-first approach
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                # Only match if distance is reasonable (< 100px)
                if D[row, col] > 100:
                    continue

                obj_id = obj_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.bboxes[obj_id] = input_bboxes[col]
                self.disappeared[obj_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            unused_rows = set(range(len(obj_centroids))) - used_rows
            for row in unused_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            # Handle unmatched new detections
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])

        return {oid: (self.objects[oid], self.bboxes[oid]) for oid in self.objects}


# ══════════════════════════════════════════════════════════
#  MODULE STATE
# ══════════════════════════════════════════════════════════
person_tracker = CentroidTracker()
bag_tracker = CentroidTracker()

# bag_id -> {'owner_id': int, 'abandon_start': float|None, 'alerted': bool}
bag_ownership = {}

# person_id -> {'positions': [(cx, cy, time), ...], 'owned_bags': [bag_id, ...]}
person_history = {}

# Track behavioral events for display
active_behaviors = {}  # person_id -> [list of behavior strings]


def reset_state():
    """Reset all tracking state (call when starting a new video feed)."""
    global person_tracker, bag_tracker, bag_ownership, person_history, active_behaviors
    person_tracker = CentroidTracker()
    bag_tracker = CentroidTracker()
    bag_ownership = {}
    person_history = {}
    active_behaviors = {}


def _centroid_from_bbox(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _detect_objects(frame):
    """Run YOLO on frame. Returns separate lists for persons and bags."""
    persons = []  # [(centroid, bbox), ...]
    bags = []

    if yolo_model is None:
        return persons, bags

    results = yolo_model(frame, verbose=False, classes=ALL_DETECT_CLASSES, conf=0.4)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centroid = _centroid_from_bbox(x1, y1, x2, y2)
            bbox = (x1, y1, x2, y2)

            if cls == PERSON_CLASS:
                persons.append((centroid, bbox))
            elif cls in BAG_CLASSES:
                bags.append((centroid, bbox))

    return persons, bags


def _associate_bags_with_persons(tracked_persons, tracked_bags):
    """Associate each new (unowned) bag with the nearest person."""
    global bag_ownership

    for bag_id, (bag_centroid, _) in tracked_bags.items():
        # Skip bags that already have an owner
        if bag_id in bag_ownership:
            continue

        # Find nearest person
        best_person_id = None
        best_dist = float('inf')

        for person_id, (person_centroid, _) in tracked_persons.items():
            dist = _distance(bag_centroid, person_centroid)
            if dist < best_dist:
                best_dist = dist
                best_person_id = person_id

        if best_person_id is not None and best_dist < ASSOCIATION_DISTANCE_PX:
            bag_ownership[bag_id] = {
                'owner_id': best_person_id,
                'abandon_start': None,
                'alerted': False
            }
            # Track that this person owns this bag
            if best_person_id not in person_history:
                person_history[best_person_id] = {'positions': [], 'owned_bags': []}
            if bag_id not in person_history[best_person_id]['owned_bags']:
                person_history[best_person_id]['owned_bags'].append(bag_id)


def _update_person_history(tracked_persons):
    """Record person positions for behavioral analysis."""
    current_time = time.time()

    for person_id, (centroid, _) in tracked_persons.items():
        if person_id not in person_history:
            person_history[person_id] = {'positions': [], 'owned_bags': []}

        person_history[person_id]['positions'].append((centroid[0], centroid[1], current_time))

        # Keep only the last 5 seconds of history
        cutoff = current_time - 10
        person_history[person_id]['positions'] = [
            p for p in person_history[person_id]['positions'] if p[2] > cutoff
        ]


def _analyze_behavior(person_id, tracked_persons):
    """
    Analyze behavioral signals for a person.
    Returns list of detected behaviors.
    """
    behaviors = []

    if person_id not in person_history:
        return behaviors

    history = person_history[person_id]
    positions = history['positions']

    if len(positions) < 5:
        return behaviors

    # ── 1. Detect "run_away" — high speed after leaving bag ──
    recent = positions[-5:]
    if len(recent) >= 2:
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        time_delta = recent[-1][2] - recent[0][2]

        if time_delta > 0:
            speed = np.sqrt(dx ** 2 + dy ** 2) / (time_delta * 30)  # pixels per frame (approx 30fps)

            if speed > RUN_SPEED_THRESHOLD:
                # Check if this person owns a bag they've left
                owned_bags = history.get('owned_bags', [])
                for bag_id in owned_bags:
                    if bag_id in bag_ownership and bag_ownership[bag_id]['abandon_start'] is not None:
                        behaviors.append("run_away")
                        break

    # ── 2. Detect "loitering" — staying in one small area for too long ──
    if len(positions) > 10:
        # Check positions over the last LOITER_TIME_SEC
        cutoff = time.time() - LOITER_TIME_SEC
        loiter_positions = [p for p in positions if p[2] > cutoff]

        if len(loiter_positions) > 20:
            # Calculate spread
            xs = [p[0] for p in loiter_positions]
            ys = [p[1] for p in loiter_positions]
            spread = max(max(xs) - min(xs), max(ys) - min(ys))

            if spread < LOITER_RADIUS_PX:
                behaviors.append("loitering")

    # ── 3. Detect "leave_object" — person was near bag, now moving away ──
    owned_bags = history.get('owned_bags', [])
    for bag_id in owned_bags:
        if bag_id in bag_ownership and bag_ownership[bag_id]['abandon_start'] is not None:
            behaviors.append("leave_object")
            break

    return behaviors


def _check_abandonment(tracked_persons, tracked_bags):
    """
    Check each tracked bag: has its owner moved away?
    Returns list of threat strings.
    """
    global bag_ownership, active_behaviors
    threats = []
    current_time = time.time()

    # Clean up ownership for bags that are no longer tracked
    for bag_id in list(bag_ownership.keys()):
        if bag_id not in tracked_bags:
            del bag_ownership[bag_id]

    for bag_id, (bag_centroid, bag_bbox) in tracked_bags.items():
        if bag_id not in bag_ownership:
            continue

        ownership = bag_ownership[bag_id]
        owner_id = ownership['owner_id']

        # Check if owner is still being tracked
        if owner_id in tracked_persons:
            owner_centroid = tracked_persons[owner_id][0]
            dist = _distance(bag_centroid, owner_centroid)

            if dist > ABANDON_DISTANCE_PX:
                # Owner has moved away from bag
                if ownership['abandon_start'] is None:
                    ownership['abandon_start'] = current_time

                elapsed = current_time - ownership['abandon_start']

                if elapsed >= ABANDON_TIME_SEC and not ownership['alerted']:
                    # ── FULL ALERT: bag abandoned ──
                    threats.append(f"Abandoned Object Alert: Bag left unattended for {elapsed:.0f}s (Owner moved away)")
                    ownership['alerted'] = True

                    # Analyze behavior of the owner
                    behaviors = _analyze_behavior(owner_id, tracked_persons)
                    active_behaviors[owner_id] = behaviors

                    if "run_away" in behaviors:
                        threats.append("Suspicious Behavior: Person RUNNING AWAY after leaving object")
                    if "loitering" in behaviors:
                        threats.append("Suspicious Behavior: Person was LOITERING before leaving object")
            else:
                # Owner is back near the bag — cancel abandonment
                ownership['abandon_start'] = None
                ownership['alerted'] = False
                if owner_id in active_behaviors:
                    del active_behaviors[owner_id]
        else:
            # Owner has completely disappeared from frame
            if ownership['abandon_start'] is None:
                ownership['abandon_start'] = current_time

            elapsed = current_time - ownership['abandon_start']

            if elapsed >= ABANDON_TIME_SEC and not ownership['alerted']:
                threats.append(f"Abandoned Object Alert: Owner left the scene, bag unattended for {elapsed:.0f}s")
                ownership['alerted'] = True

    return threats


def _draw_annotations(frame, tracked_persons, tracked_bags):
    """Draw all visual annotations on the frame."""
    current_time = time.time()

    # ── Draw persons ──
    for person_id, (centroid, bbox) in tracked_persons.items():
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)  # Green by default

        # Check if this person has any active behavioral alerts
        behaviors = active_behaviors.get(person_id, [])
        label = f"Person #{person_id}"

        if "run_away" in behaviors:
            color = (0, 0, 255)  # Red
            label += " [RUNNING AWAY!]"
        elif "leave_object" in behaviors:
            color = (0, 140, 255)  # Orange
            label += " [LEFT OBJECT]"
        elif "loitering" in behaviors:
            color = (0, 255, 255)  # Yellow
            label += " [LOITERING]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ── Draw bags ──
    for bag_id, (centroid, bbox) in tracked_bags.items():
        x1, y1, x2, y2 = bbox
        ownership = bag_ownership.get(bag_id, None)

        if ownership is None:
            # Unassociated bag — gray
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
            cv2.putText(frame, f"Bag #{bag_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        elif ownership['alerted']:
            # ── ABANDONED — Red pulsing ──
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), 3)
            elapsed = current_time - (ownership['abandon_start'] or current_time)
            cv2.putText(frame, f"ABANDONED ({elapsed:.0f}s)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Flashing warning overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            alpha = 0.15 + 0.1 * np.sin(current_time * 5)  # Pulsing effect
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        elif ownership['abandon_start'] is not None:
            # ── UNATTENDED — Orange warning ──
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            elapsed = current_time - ownership['abandon_start']
            remaining = max(0, ABANDON_TIME_SEC - elapsed)
            cv2.putText(frame, f"UNATTENDED ({remaining:.0f}s)", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        else:
            # ── Bag with owner nearby — Yellow ──
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Bag #{bag_id} (Owner #{ownership['owner_id']})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Draw line connecting bag to owner
            owner_id = ownership['owner_id']
            if owner_id in tracked_persons:
                owner_centroid = tracked_persons[owner_id][0]
                cv2.line(frame, centroid, owner_centroid, (255, 255, 0), 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════
def process_frame_for_abandoned_objects(frame):
    """
    Enhanced abandoned object detection with person-bag tracking.

    Pipeline:
      1. YOLO detects persons and bags
      2. CentroidTracker assigns persistent IDs
      3. Bags are associated with nearest person (owner)
      4. If owner moves away > threshold → start unattended timer
      5. If bag unattended > X seconds → ALERT
      6. Behavioral analysis: leave_object, run_away, loitering

    Returns: (annotated_frame, list_of_threat_strings)
    """
    threats = []

    # Step 1: Detect persons and bags
    person_detections, bag_detections = _detect_objects(frame)

    # Step 2: Update trackers
    tracked_persons = person_tracker.update(person_detections)
    tracked_bags = bag_tracker.update(bag_detections)

    # Step 3: Update person movement history
    _update_person_history(tracked_persons)

    # Step 4: Associate bags with nearest persons
    _associate_bags_with_persons(tracked_persons, tracked_bags)

    # Step 5: Check abandonment + behavioral analysis
    abandon_threats = _check_abandonment(tracked_persons, tracked_bags)
    threats.extend(abandon_threats)

    # Step 6: Draw annotations
    frame = _draw_annotations(frame, tracked_persons, tracked_bags)

    # Step 7: Draw status overlay
    status_text = f"Tracking: {len(tracked_persons)} persons, {len(tracked_bags)} bags"
    cv2.putText(frame, status_text, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame, threats
