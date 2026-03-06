import streamlit as st
import cv2
import time
import os
import json
from datetime import datetime
from modules.face_recognition_module import process_frame_for_faces
from modules.weapon_detection_module import process_frame_for_weapons, reset_state as reset_weapon
from modules.activity_detection_module import process_frame_for_activity, reset_state as reset_activity
from modules.abandoned_object_module import process_frame_for_abandoned_objects, reset_state as reset_abandoned
from modules.traffic_accident_module import process_frame_for_accidents, reset_state as reset_traffic
from modules.alerts import check_and_send_alerts
from modules.training_pipeline import render_training_ui

# ══════════════════════════════════════════════════════════
#  Streamlit Page Config
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="UrbanDefense AI", page_icon="🛡️", layout="wide")

# ══════════════════════════════════════════════════════════
#  Premium Glassmorphism CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Global ─── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main {
    background: linear-gradient(135deg, #0a0e27 0%, #0d1537 40%, #111d4a 70%, #0a1628 100%);
}
header[data-testid="stHeader"] {
    background: rgba(10, 14, 39, 0.8);
    backdrop-filter: blur(12px);
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1537 0%, #162055 50%, #0d1537 100%);
    border-right: 1px solid rgba(76, 201, 240, 0.15);
}
section[data-testid="stSidebar"] .stRadio label {
    color: #b8c5d6 !important;
    font-weight: 500;
}

/* ─── Cards / Glass Panels ─── */
.glass-card {
    background: rgba(22, 32, 85, 0.45);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(76, 201, 240, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.glass-card-alert {
    background: rgba(255, 59, 48, 0.12);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 59, 48, 0.3);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #ff6b6b;
    font-weight: 500;
}
.glass-card-info {
    background: rgba(76, 201, 240, 0.08);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(76, 201, 240, 0.2);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #7dd3fc;
    font-weight: 400;
}

/* ─── Headings ─── */
h1 {
    background: linear-gradient(90deg, #4cc9f0, #7b2ff7, #f72585);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
h2, h3 {
    color: #4cc9f0 !important;
    font-weight: 600 !important;
}

/* ─── Metrics ─── */
[data-testid="stMetric"] {
    background: rgba(22, 32, 85, 0.5);
    border: 1px solid rgba(76, 201, 240, 0.15);
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricLabel"] {
    color: #7dd3fc !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #4cc9f0 0%, #7b2ff7 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(76, 201, 240, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(76, 201, 240, 0.5);
}

/* ─── Checkbox / Inputs ─── */
.stCheckbox label span {
    color: #b8c5d6 !important;
}
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: rgba(22, 32, 85, 0.6) !important;
    border: 1px solid rgba(76, 201, 240, 0.2) !important;
    border-radius: 8px !important;
    color: #e0e7ff !important;
}

/* ─── Detection Log ─── */
.detection-log {
    background: rgba(13, 21, 55, 0.7);
    border: 1px solid rgba(76, 201, 240, 0.15);
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
    max-height: 200px;
    overflow-y: auto;
}
.detection-item {
    display: flex;
    align-items: center;
    padding: 6px 10px;
    margin: 4px 0;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
}
.detection-item.danger {
    background: rgba(255, 59, 48, 0.15);
    border-left: 3px solid #ff3b30;
    color: #ff6b6b;
}
.detection-item.safe {
    background: rgba(52, 199, 89, 0.1);
    border-left: 3px solid #34c759;
    color: #6ee7a0;
}

/* ─── Status Badge ─── */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.status-active {
    background: rgba(52, 199, 89, 0.2);
    color: #34c759;
    border: 1px solid rgba(52, 199, 89, 0.3);
}
.status-idle {
    background: rgba(255, 214, 10, 0.15);
    color: #ffd60a;
    border: 1px solid rgba(255, 214, 10, 0.3);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="text-align:center; padding: 10px 0 20px 0;">
    <div style="font-size: 42px;">🛡️</div>
    <h2 style="margin:0; background: linear-gradient(90deg, #4cc9f0, #7b2ff7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 22px;">UrbanDefense AI</h2>
    <p style="color: #7dd3fc; font-size: 12px; margin:4px 0 0 0; letter-spacing: 1px;">
        SMART CITY CRIME PREVENTION</p>
</div>
""", unsafe_allow_html=True)

menu = ["🎥 Live Monitoring", "📜 Alerts History", "🗺️ Threat Map", "⚙️ Configuration", "🧠 Model Training"]
choice = st.sidebar.radio("Navigation", menu, label_visibility="collapsed")

# ── Ensure directories ──
for d in ["Datasets/Known_Faces", "Datasets/Weapons", "Datasets/Activities",
          "Datasets/Abandoned_Objects", "Datasets/Accidents", "Data"]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════
#  🎥 LIVE MONITORING
# ══════════════════════════════════════════════════════════
if choice == "🎥 Live Monitoring":
    st.markdown('<h1>🎥 Live Surveillance Monitoring</h1>', unsafe_allow_html=True)

    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔧 Detection Modules")
        enable_face = st.checkbox("👤 Face Recognition", value=False)
        enable_weapon = st.checkbox("🔫 Weapon Detection", value=True)
        enable_activity = st.checkbox("🏃 Suspicious Activity", value=False)
        enable_abandoned = st.checkbox("🎒 Abandoned Object", value=True)
        enable_accident = st.checkbox("🚗 Traffic Accident", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🚨 Live Alerts")
        alert_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_main:
        frame_placeholder = st.empty()
        # Detection Log area — BELOW the camera feed
        st.markdown("### 📋 Detection Log (Real-Time)")
        detection_log_placeholder = st.empty()

    # Sidebar controls
    st.sidebar.markdown("---")
    start_button = st.sidebar.button("▶️ Start Camera / Video")
    stop_button = st.sidebar.button("⏹️ Stop")
    uploaded_video = st.sidebar.file_uploader("📁 Upload a Video", type=["mp4", "avi", "mov"])

    # Frame skip setting for performance
    st.sidebar.markdown("---")
    frame_skip = st.sidebar.slider("⚡ Process every Nth frame", 1, 10, 3,
                                    help="Higher = faster but fewer detections. Set to 1 for max accuracy.")

    if start_button:
        video_source = 0
        if uploaded_video is not None:
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.read())
            video_source = temp_path

        cap = cv2.VideoCapture(video_source)
        st.session_state['run_video'] = True
        
        # Reset all detector states for fresh start
        reset_activity()
        reset_abandoned()
        reset_traffic()
        reset_weapon()

        frame_count = 0
        detection_history = []  # Keep a running log

        while cap.isOpened() and st.session_state.get('run_video', False):
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            active_threats = []

            # ── Only run heavy AI models every Nth frame ──
            run_detection = (frame_count % frame_skip == 0)

            if run_detection:
                if enable_face:
                    frame, threats = process_frame_for_faces(frame)
                    active_threats.extend(threats)

                if enable_weapon:
                    frame, threats = process_frame_for_weapons(frame)
                    active_threats.extend(threats)

                if enable_activity:
                    frame, threats = process_frame_for_activity(frame)
                    active_threats.extend(threats)

                if enable_abandoned:
                    frame, threats = process_frame_for_abandoned_objects(frame)
                    active_threats.extend(threats)

                if enable_accident:
                    frame, threats = process_frame_for_accidents(frame)
                    active_threats.extend(threats)

                # Send SMS alerts for genuine threats (pass frame for image capture)
                if active_threats:
                    check_and_send_alerts(active_threats, frame=frame)
                    for t in active_threats:
                        detection_history.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "type": t
                        })

            # ── Display frame ──
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # ── Live Alerts sidebar ──
            if active_threats:
                with alert_placeholder.container():
                    for t in active_threats[-5:]:
                        st.markdown(f'<div class="glass-card-alert">🚨 {t}</div>', unsafe_allow_html=True)
            else:
                if run_detection:
                    alert_placeholder.markdown(
                        '<div class="glass-card-info">✅ No threats detected</div>',
                        unsafe_allow_html=True)

            # ── Detection Log below camera ──
            if detection_history:
                with detection_log_placeholder.container():
                    st.markdown('<div class="detection-log">', unsafe_allow_html=True)
                    for entry in reversed(detection_history[-15:]):
                        st.markdown(
                            f'<div class="detection-item danger">'
                            f'<span style="margin-right:8px;">⏱ {entry["time"]}</span> '
                            f'{entry["type"]}</div>',
                            unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                detection_log_placeholder.markdown(
                    '<div class="detection-log">'
                    '<div class="detection-item safe">No detections yet — waiting for camera data…</div>'
                    '</div>', unsafe_allow_html=True)

            time.sleep(0.03)

        cap.release()

    if stop_button:
        st.session_state['run_video'] = False
        st.sidebar.warning("Processing Stopped.")

# ══════════════════════════════════════════════════════════
#  📜 ALERTS HISTORY
# ══════════════════════════════════════════════════════════
elif choice == "📜 Alerts History":
    st.markdown('<h1>📜 Alerts History</h1>', unsafe_allow_html=True)
    st.markdown("Historical log of all detected threats and dispatched alerts.")

    if os.path.exists("Data/alerts.log"):
        with open("Data/alerts.log", "r") as f:
            logs = f.readlines()

        if logs:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            for log in reversed(logs[-30:]):
                st.markdown(f'<div class="glass-card-alert">{log.strip()}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🗑️ Clear Alert History"):
                os.remove("Data/alerts.log")
                st.success("Alert history cleared!")
                st.rerun()
        else:
            st.info("No recorded alerts yet.")
    else:
        st.info("No recorded alerts yet. Alerts will appear here once detections are triggered.")

# ══════════════════════════════════════════════════════════
#  🗺️ THREAT MAP
# ══════════════════════════════════════════════════════════
elif choice == "🗺️ Threat Map":
    st.markdown('<h1>🗺️ High-Crime Threat Map</h1>', unsafe_allow_html=True)
    st.markdown("Interactive map highlighting areas with unusual activity based on recent camera alerts.")

    try:
        import folium
        from streamlit_folium import st_folium
        import pandas as pd

        # ── Stats Row ──
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚨 Total Alerts", "12", "+3 today")
        with col2:
            st.metric("🔴 Critical Zones", "2", "Sector A, D")
        with col3:
            st.metric("🟡 Medium Zones", "3", "Sector B, C, E")
        with col4:
            st.metric("🟢 Safe Zones", "5", "All Clear")

        st.markdown("---")

        # ── Map Data (mock demo data — replace with real coordinates) ──
        crime_data = {
            'lat': [17.3850, 17.3950, 17.4050, 17.3750, 17.3650, 17.4150, 17.3550],
            'lon': [78.4867, 78.4767, 78.4967, 78.5067, 78.4567, 78.5167, 78.4667],
            'crime_type': [
                'Abandoned Bag', 'Crowd Fight', 'Weapon Spotted', 
                'Traffic Accident', 'Suspicious Activity', 'Theft Detected', 'Unusual Crowd'
            ],
            'severity': ['High', 'Critical', 'Critical', 'High', 'Medium', 'Medium', 'Low'],
            'color': ['orange', 'red', 'darkred', 'orange', 'yellow', 'yellow', 'green'],
            'radius': [18, 25, 25, 18, 12, 12, 8]
        }
        df = pd.DataFrame(crime_data)

        # Build the folium map (dark theme)
        m = folium.Map(
            location=[17.3850, 78.4867],
            zoom_start=13,
            tiles="CartoDB dark_matter"
        )

        # Add heatmap-style circle markers
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=row['radius'],
                popup=folium.Popup(
                    f"<div style='font-family:Inter;'>"
                    f"<b style='color:#ff3b30;'>{row['crime_type']}</b><br>"
                    f"Severity: <b>{row['severity']}</b><br>"
                    f"Time: {datetime.now().strftime('%H:%M')}</div>",
                    max_width=200
                ),
                color=row['color'],
                fill=True,
                fill_color=row['color'],
                fill_opacity=0.6,
                weight=2
            ).add_to(m)

        # Render in Streamlit
        st_folium(m, width=None, height=550, use_container_width=True)

        # Legend
        st.markdown("""
        <div class="glass-card" style="margin-top: 16px;">
            <h3 style="margin-top:0;">Legend</h3>
            <span style="color:#ff3b30;">🔴 Critical</span> — Weapons, active fights &nbsp;&nbsp;
            <span style="color:#ff9500;">🟠 High</span> — Abandoned objects, accidents &nbsp;&nbsp;
            <span style="color:#ffd60a;">🟡 Medium</span> — Suspicious activity, theft &nbsp;&nbsp;
            <span style="color:#34c759;">🟢 Low</span> — Unusual crowd, monitored
        </div>
        """, unsafe_allow_html=True)

    except ImportError:
        st.error("Map libraries not installed. Run: `pip install folium streamlit-folium pandas`")

# ══════════════════════════════════════════════════════════
#  ⚙️ CONFIGURATION
# ══════════════════════════════════════════════════════════
elif choice == "⚙️ Configuration":
    st.markdown('<h1>⚙️ System Configuration</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📱 Alert Settings (Twilio)")
    st.caption("Enter your Twilio credentials to enable unlimited SMS alerts.")
    
    # Try to load existing config
    existing_phone = "6301025618"
    existing_sid = ""
    existing_token = ""
    existing_twilio_num = ""
    
    try:
        import json
        import os
        if os.path.exists("Data/config.json"):
            with open("Data/config.json", "r") as f:
                cfg = json.load(f)
                existing_phone = cfg.get("PHONE_NUMBER", existing_phone)
                existing_sid = cfg.get("TWILIO_SID", "")
                existing_token = cfg.get("TWILIO_TOKEN", "")
                existing_twilio_num = cfg.get("TWILIO_NUM", "")
    except Exception:
        pass

    target_number = st.text_input("Helpline Phone Number (To)", value=existing_phone)
    twil_sid = st.text_input("Twilio Account SID", value=existing_sid)
    twil_token = st.text_input("Twilio Auth Token", value=existing_token, type="password")
    twil_num = st.text_input("Your Twilio Phone Number (From)", value=existing_twilio_num)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Location Settings ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📍 Camera / Device Location (GPS)")
    st.caption("Auto-detect your device location or enter coordinates manually.")
    
    existing_location = cfg.get("CAMERA_LOCATION", "Main Gate - Block A") if os.path.exists("Data/config.json") else "Main Gate - Block A"
    existing_lat = cfg.get("CAMERA_LATITUDE", 0.0) if os.path.exists("Data/config.json") else 0.0
    existing_lon = cfg.get("CAMERA_LONGITUDE", 0.0) if os.path.exists("Data/config.json") else 0.0
    existing_address = cfg.get("CAMERA_ADDRESS", "") if os.path.exists("Data/config.json") else ""
    
    # ── Check if browser sent GPS coordinates via query params ──
    query_params = st.query_params
    if "dev_lat" in query_params and "dev_lon" in query_params:
        try:
            detected_lat = float(query_params["dev_lat"])
            detected_lon = float(query_params["dev_lon"])
            st.session_state["device_lat"] = detected_lat
            st.session_state["device_lon"] = detected_lon
            # Reverse geocode the detected location
            try:
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="urbandefense_ai")
                loc_result = geolocator.reverse(f"{detected_lat}, {detected_lon}", exactly_one=True, language="en")
                if loc_result and loc_result.address:
                    st.session_state["fetched_address"] = loc_result.address
            except Exception:
                pass
            # Clear query params so it doesn't re-trigger
            st.query_params.clear()
        except (ValueError, TypeError):
            pass
    
    # Use device-detected location if available, else existing config
    use_lat = st.session_state.get("device_lat", existing_lat)
    use_lon = st.session_state.get("device_lon", existing_lon)
    
    cam_location = st.text_input("Camera Location Name", value=existing_location,
                                  help="A friendly name like 'Main Gate - Block A'")
    
    # ── Auto-detect device location button ──
    import streamlit.components.v1 as components
    
    st.markdown("#### 🛰️ Auto-Detect Device Location")
    detect_col1, detect_col2 = st.columns([1, 2])
    with detect_col1:
        detect_clicked = st.button("📍 Detect My Device Location", use_container_width=True)
    with detect_col2:
        if st.session_state.get("device_lat"):
            st.success(f"✅ Device located: {st.session_state['device_lat']:.6f}, {st.session_state['device_lon']:.6f}")
    
    if detect_clicked:
        # Inject JavaScript to get browser geolocation and redirect back with coords
        geolocation_js = """
        <script>
        function getDeviceLocation() {
            const statusEl = document.getElementById('geo-status');
            if (!navigator.geolocation) {
                statusEl.innerHTML = '❌ Geolocation is not supported by your browser.';
                return;
            }
            statusEl.innerHTML = '⏳ Detecting your location... Please allow location access in your browser.';
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    const accuracy = position.coords.accuracy;
                    statusEl.innerHTML = '✅ Location detected! Lat: ' + lat.toFixed(6) + ', Lon: ' + lon.toFixed(6) + ' (Accuracy: ' + accuracy.toFixed(0) + 'm). Updating...';
                    // Redirect to same page with coordinates as query params
                    const currentUrl = new URL(window.parent.location.href);
                    currentUrl.searchParams.set('dev_lat', lat.toFixed(8));
                    currentUrl.searchParams.set('dev_lon', lon.toFixed(8));
                    setTimeout(function() {
                        window.parent.location.href = currentUrl.toString();
                    }, 1000);
                },
                function(error) {
                    let msg = '❌ Location access denied. ';
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            msg += 'Please enable location permissions in your browser settings.';
                            break;
                        case error.POSITION_UNAVAILABLE:
                            msg += 'Location information is unavailable on this device.';
                            break;
                        case error.TIMEOUT:
                            msg += 'The request timed out. Please try again.';
                            break;
                    }
                    statusEl.innerHTML = msg;
                },
                {
                    enableHighAccuracy: true,
                    timeout: 15000,
                    maximumAge: 0
                }
            );
        }
        // Auto-trigger on load
        getDeviceLocation();
        </script>
        <div id="geo-status" style="color: #7dd3fc; font-family: Inter, sans-serif; padding: 10px; 
             background: rgba(76,201,240,0.08); border: 1px solid rgba(76,201,240,0.2); 
             border-radius: 8px; margin: 5px 0;">
            ⏳ Requesting device location...
        </div>
        """
        components.html(geolocation_js, height=60)
    
    st.markdown("#### 📐 GPS Coordinates")
    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        cam_lat = st.number_input("Latitude", value=float(use_lat), format="%.6f",
                                   min_value=-90.0, max_value=90.0, step=0.0001)
    with loc_col2:
        cam_lon = st.number_input("Longitude", value=float(use_lon), format="%.6f",
                                   min_value=-180.0, max_value=180.0, step=0.0001)
    
    # Reverse geocode button
    if st.button("🔍 Get Address from Coordinates"):
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="urbandefense_ai")
            location_result = geolocator.reverse(f"{cam_lat}, {cam_lon}", exactly_one=True, language="en")
            if location_result and location_result.address:
                st.session_state['fetched_address'] = location_result.address
                st.success(f"📍 Address: {location_result.address}")
            else:
                st.warning("Could not find an address for these coordinates.")
        except Exception as e:
            st.error(f"Geocoding failed: {e}. Make sure `geopy` is installed: `pip install geopy`")
    
    # Use fetched address or existing
    addr_default = st.session_state.get('fetched_address', existing_address)
    cam_address = st.text_input("Full Address (auto-filled or manual)", value=addr_default,
                                 help="Click 'Detect My Device Location' or 'Get Address' to auto-fill.")
    
    if cam_lat and cam_lon and (cam_lat != 0.0 or cam_lon != 0.0):
        st.markdown(f'<div class="glass-card-info">🗺️ <a href="https://maps.google.com/maps?q={cam_lat},{cam_lon}" target="_blank" style="color:#7dd3fc;">View on Google Maps</a></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Module Sensitivities")
    weapon_thresh = st.slider("Weapon Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    face_tol = st.slider("Face Match Tolerance", 0.1, 1.0, 0.6, 0.05)
    activity_thresh = st.slider("Activity Motion Threshold", 10000, 100000, 50000, 5000,
                                 help="Higher = fewer false positives. Lower = more sensitive.")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("💾 Save Configuration"):
        # Auto-fetch address on save if not already provided
        if not cam_address and cam_lat and cam_lon:
            try:
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="urbandefense_ai")
                location_result = geolocator.reverse(f"{cam_lat}, {cam_lon}", exactly_one=True, language="en")
                if location_result and location_result.address:
                    cam_address = location_result.address
            except Exception:
                pass
        
        config = {
            "PHONE_NUMBER": target_number,
            "TWILIO_SID": twil_sid,
            "TWILIO_TOKEN": twil_token,
            "TWILIO_NUM": twil_num,
            "CAMERA_LOCATION": cam_location,
            "CAMERA_LATITUDE": cam_lat,
            "CAMERA_LONGITUDE": cam_lon,
            "CAMERA_ADDRESS": cam_address,
            "WEAPON_THRESHOLD": weapon_thresh,
            "FACE_TOLERANCE": face_tol,
            "ACTIVITY_THRESHOLD": activity_thresh
        }
        with open("Data/config.json", "w") as f:
            json.dump(config, f, indent=2)
        st.success("✅ Configuration saved with GPS location!")

# ══════════════════════════════════════════════════════════
#  🧠 MODEL TRAINING
# ══════════════════════════════════════════════════════════
elif choice == "🧠 Model Training":
    st.markdown('<h1>🧠 Model Training Pipeline</h1>', unsafe_allow_html=True)
    render_training_ui()
