import os
import shutil
from ultralytics import YOLO

def main():
    print("Starting automated YOLOv8 training for Weapon Detection...")
    
    # Define paths
    yaml_path = r"e:\Criminal-Activity-Video-Surveillance-using-Deep-Learning-main\Datasets\Weapons\gun\data.yaml"
    model_save_path = r"e:\Criminal-Activity-Video-Surveillance-using-Deep-Learning-main\Models\weapon_model.pt"
    
    # Load base model
    model = YOLO("yolov8s.pt") # Using small model for better accuracy than nano
    
    try:
        # Train the model (50 epochs, imgsz 640)
        results = model.train(
            data=yaml_path,
            epochs=50,
            imgsz=640,
            batch=16,
            device='', # Auto-detect GPU/CPU
            plots=True,
            exist_ok=True
        )
        
        # After training, the best model is saved here
        best_pt_path = r"e:\Criminal-Activity-Video-Surveillance-using-Deep-Learning-main\runs\detect\train\weights\best.pt"
        
        if os.path.exists(best_pt_path):
            print(f"Training complete! Moving best model to {model_save_path}")
            shutil.copy(best_pt_path, model_save_path)
            print("Successfully moved! Weapon detection is now fully active.")
            print("You can safely close this terminal/window.")
        else:
            print("Error: Could not find best.pt after training.")
            
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
