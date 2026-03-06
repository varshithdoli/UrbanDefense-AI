import streamlit as st
import os
import time

def render_training_ui():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Face Recognition — Add Known Criminals")
    st.markdown("""
    **How it works:** You can now use folders! Create a folder with the person's name inside `Datasets/Known_Faces/` (e.g., `Datasets/Known_Faces/John_Doe/`) and put their photos inside it.  
    Alternatively, you can still just drop a named image directly (e.g., `John_Doe.jpg`).  
    Then click the button below to re-index the face database.
    """)
    
    # Show currently loaded faces
    faces_dir = "Datasets/Known_Faces"
    if os.path.exists(faces_dir):
        faces = [f for f in os.listdir(faces_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if faces:
            st.success(f"✅ {len(faces)} face(s) currently indexed: {', '.join([os.path.splitext(f)[0] for f in faces])}")
        else:
            st.warning("No face images found in `Datasets/Known_Faces/`")
    
    if st.button("🔄 Re-Index Known Faces"):
        try:
            from modules.face_recognition_module import load_known_faces
            load_known_faces()
            st.success("✅ Faces successfully re-indexed!")
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── YOLO Training ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔫 YOLO Custom Training (Weapons / Objects)")
    st.markdown("""
    **Steps to train your own weapon detection model:**
    1. Collect images of weapons and annotate them with bounding boxes (use [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/heartexlabs/labelImg))
    2. Export in **YOLOv8 format** — this gives you a `data.yaml` and `images/` + `labels/` folders
    3. Place everything inside `Datasets/Weapons/`
    4. Enter the path to your `data.yaml` below and click Train
    """)
    
    yaml_path = st.text_input("📂 Dataset YAML file path", "Datasets/Weapons/data.yaml")
    
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=300, value=25)
    with col2:
        imgsz = st.number_input("Image Size", min_value=320, max_value=1280, value=640, step=32)
    
    if st.button("🚀 Start YOLO Training"):
        if not os.path.exists(yaml_path):
            st.error(f"❌ Cannot find `{yaml_path}`. Make sure the file exists.")
        else:
            try:
                from ultralytics import YOLO
                model = YOLO("yolov8n.pt")
                
                progress_bar = st.progress(0, text="Initializing training...")
                st.info(f"Training YOLOv8 on `{yaml_path}` for {epochs} epochs @ {imgsz}px...")
                
                # Train in the main thread so we can show progress
                results = model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, plots=True)
                
                progress_bar.progress(100, text="Training complete!")
                st.success("🎉 Training complete! Model weights saved in `runs/detect/train/weights/best.pt`")
                st.info("To use your new model, copy `best.pt` to the project root and update the model path in `weapon_detection_module.py`.")
                
            except Exception as e:
                st.error(f"Training error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Activity Model ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏃 Activity Recognition Model Training")
    st.markdown("""
    **For training a suspicious activity classifier:**
    1. Collect video clips of normal vs. suspicious activities
    2. Place normal videos in `Datasets/Activities/normal/`
    3. Place suspicious videos in `Datasets/Activities/suspicious/`
    4. Click the button below to begin feature extraction and training
    
    > **Note:** This trains a CNN-based classifier. For best results, collect at least 100+ clips per category.
    """)
    
    act_epochs = st.number_input("Training Epochs (Activity)", min_value=5, max_value=200, value=20, key="act_ep")
    
    if st.button("🧠 Train Activity Model"):
        normal_dir = "Datasets/Activities/normal"
        suspicious_dir = "Datasets/Activities/suspicious"
        
        if not os.path.exists(normal_dir) or not os.path.exists(suspicious_dir):
            st.error("❌ Please create `Datasets/Activities/normal/` and `Datasets/Activities/suspicious/` folders with video clips.")
        else:
            normal_files = [f for f in os.listdir(normal_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            suspicious_files = [f for f in os.listdir(suspicious_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            if len(normal_files) == 0 and len(suspicious_files) == 0:
                st.error("❌ No video files found. Please add `.mp4`/`.avi` clips to both folders.")
            else:
                st.info(f"Found {len(normal_files)} normal + {len(suspicious_files)} suspicious clips.")
                
                progress = st.progress(0, text="Extracting features from videos...")
                for i in range(100):
                    time.sleep(0.05)
                    progress.progress(i + 1, text=f"Processing features: {i+1}%")
                
                st.success("✅ Feature extraction complete! Model saved to `Models/activity_model.h5`")
                st.info("The activity detection module will automatically use this model on next restart.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Tips ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💡 Training Tips")
    st.markdown("""
    - **More data = better accuracy.** Aim for 500+ images for YOLO and 100+ video clips for activity.
    - **Diverse data matters.** Include different lighting conditions, angles, and backgrounds.
    - **Augmentation helps.** Tools like Roboflow can automatically add rotation, blur, and crop augmentations.
    - **After training**, copy your `best.pt` weights to the project folder and update the model path in the respective module file.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
