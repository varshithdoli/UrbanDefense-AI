import cv2
import face_recognition
import os
import numpy as np

# Load known faces from Datasets/Known_Faces
KNOWN_FACES_DIR = "Datasets/Known_Faces"

known_encodings = []
known_names = []

def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Warning: {KNOWN_FACES_DIR} does not exist.")
        return
        
    print(f"Loading faces from {KNOWN_FACES_DIR}...")
    
    # We use os.walk to support both:
    # 1. Direct files: Known_Faces/John.jpg
    # 2. Folder structure: Known_Faces/John/1.jpg
    for root, dirs, files in os.walk(KNOWN_FACES_DIR):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                normalized_root = os.path.normpath(root)
                normalized_base = os.path.normpath(KNOWN_FACES_DIR)
                
                # If file is directly in Known_Faces, use filename as name
                # If file is in a subfolder (e.g. Known_Faces/John/img1.jpg), use folder name ('John')
                if normalized_root == normalized_base:
                    name = os.path.splitext(filename)[0]
                else:
                    name = os.path.basename(normalized_root)
                    
                img_path = os.path.join(root, filename)
                try:
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                        print(f"Loaded: {name} from {img_path}")
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    
    print(f"Loaded {len(known_encodings)} face encodings.")

# Initial load
load_known_faces()

def process_frame_for_faces(frame, tolerance=0.5):
    """
    Detect known faces. If distance is <= tolerance, consider it a match.
    Displays the name and confidence percentage.
    """
    threats = []
    if not known_encodings:
        return frame, threats

    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return frame, threats
        
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Calculate distances to all known faces
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        # Find the best match (smallest distance)
        best_match_index = np.argmin(face_distances)
        min_distance = face_distances[best_match_index]
        
        # If the best match is within tolerance
        if min_distance <= tolerance:
            name = known_names[best_match_index]
            
            # Convert distance to a confidence percentage (strictness)
            # Distance 0 is 100% confidence. Distance == tolerance is ~50% confidence.
            confidence = (1.0 - min_distance) / (1.0 - (tolerance / 2)) * 100
            confidence = min(100.0, max(0.0, confidence))
            
            threat_msg = f"Known Criminal {name} Detected (Conf: {confidence:.0f}%)"
            threats.append(threat_msg)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} {confidence:.0f}%", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            # Face found but unknown
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            pass

    return frame, threats
