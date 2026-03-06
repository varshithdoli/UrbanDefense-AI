import os
import cv2
import json
import time
import threading
import requests
from datetime import datetime

# Cooldown to avoid spamming alerts for the same threat constantly
alert_cooldowns = {}
COOLDOWN_SECONDS = 30

SNAPSHOTS_DIR = "Data/snapshots"


def get_config():
    if os.path.exists("Data/config.json"):
        try:
            with open("Data/config.json", "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_target_phone():
    cfg = get_config()
    return cfg.get("PHONE_NUMBER", "6301025618")


def save_snapshot(frame, threat_label):
    """Save detected frame to Data/snapshots/ and return the file path."""
    try:
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in threat_label)[:40]
        filename = f"{timestamp}_{safe_label}.jpg"
        filepath = os.path.join(SNAPSHOTS_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"📸 Snapshot saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Failed to save snapshot: {e}")
        return None


def upload_image_for_mms(filepath):
    """
    Upload image to imgbb (free, no API key required for anonymous uploads)
    Returns a public URL for Twilio MMS, or None on failure.
    """
    try:
        import base64
        with open(filepath, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # imgbb free anonymous upload (using free key)
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            data={
                "key": "5e6b3e56b1f8387a3610405361e0a986",  # Free public demo key
                "image": img_data,
                "expiration": 600,  # 10 minutes — enough for Twilio to fetch
            },
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            url = result.get("data", {}).get("url", None)
            if url:
                print(f"📤 Image uploaded for MMS: {url}")
                return url
        
        print(f"⚠️ Image upload returned status {response.status_code}")
        return None
    except Exception as e:
        print(f"⚠️ Image upload failed (SMS will be sent without image): {e}")
        return None


def build_alert_message(threat, cfg):
    """Build a rich SMS message with location details."""
    location_name = cfg.get("CAMERA_LOCATION", "Unknown Location")
    lat = cfg.get("CAMERA_LATITUDE", None)
    lon = cfg.get("CAMERA_LONGITUDE", None)
    address = cfg.get("CAMERA_ADDRESS", "")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg_parts = [
        f"🚨 URBAN DEFENSE ALERT",
        f"Threat: {threat}",
        f"Location: {location_name}",
    ]

    if address:
        msg_parts.append(f"Address: {address}")

    if lat is not None and lon is not None:
        msg_parts.append(f"GPS: {lat}, {lon}")
        msg_parts.append(f"Maps: https://maps.google.com/maps?q={lat},{lon}")

    msg_parts.append(f"Time: {timestamp}")

    return "\n".join(msg_parts)


def send_sms(message_body, to_number, media_url=None):
    try:
        cfg = get_config()
        sid = cfg.get("TWILIO_SID", "")
        token = cfg.get("TWILIO_TOKEN", "")
        from_num = cfg.get("TWILIO_NUM", "")

        # If user hasn't configured Twilio, fall back to console log
        if not sid or not token or not from_num:
            print(f"\n[ALERT SYSTEM] SMS Pending (Setup Twilio in Config Tab) -> {to_number}:")
            print(f"{message_body}")
            if media_url:
                print(f"📎 Image: {media_url}")
            return

        print(f"\n[ALERT SYSTEM] Sending REAL SMS/MMS to {to_number} via Twilio...")

        from twilio.rest import Client

        # Format the numbers correctly (Twilio requires +CountryCode)
        if not to_number.startswith("+"):
            to_number = f"+91{to_number}"  # Default to India if not specified

        client = Client(sid, token)

        msg_kwargs = {
            "body": message_body,
            "from_": from_num,
            "to": to_number,
        }

        # Add image as MMS if available
        if media_url:
            msg_kwargs["media_url"] = [media_url]
            print(f"📎 Attaching image: {media_url}")

        message = client.messages.create(**msg_kwargs)
        msg_type = "MMS" if media_url else "SMS"
        print(f"✅ {msg_type} successfully DELIVERED! Twilio Message SID: {message.sid}")

    except Exception as e:
        print(f"❌ Failed to send Twilio SMS/MMS: {e}")


def _send_alert_thread(threat, frame, target_number):
    """Background thread: save snapshot, upload, build message, send SMS/MMS."""
    cfg = get_config()
    alert_msg = build_alert_message(threat, cfg)

    media_url = None
    if frame is not None:
        snapshot_path = save_snapshot(frame, threat)
        if snapshot_path:
            media_url = upload_image_for_mms(snapshot_path)

    send_sms(alert_msg, target_number, media_url=media_url)


def check_and_send_alerts(threats, frame=None):
    """
    Check cooldowns and dispatch alerts for each threat.
    
    Args:
        threats: list of threat description strings
        frame: optional current video frame (numpy array) to capture snapshot
    """
    current_time = time.time()
    target_number = get_target_phone()

    for threat in threats:
        # Extract base threat type for cooldown (e.g., "Weapon Detected")
        threat_base = threat.split(":")[0]

        last_alert_time = alert_cooldowns.get(threat_base, 0)

        if current_time - last_alert_time > COOLDOWN_SECONDS:
            # Log it
            os.makedirs("Data", exist_ok=True)
            with open("Data/alerts.log", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] {threat}\n")

            # Send SMS/MMS async so it doesn't block video pipeline
            threading.Thread(
                target=_send_alert_thread,
                args=(threat, frame, target_number)
            ).start()

            alert_cooldowns[threat_base] = current_time
