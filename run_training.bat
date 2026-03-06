@echo off
title YOLOv8 Weapon Training
echo ----------------------------------------------------
echo Starting YOLOv8 Weapon Training...
echo This window will stay open to show you the progress.
echo ----------------------------------------------------
python auto_train.py
echo.
echo ----------------------------------------------------
echo Training complete! The system will now use the model.
echo You can safely close this window now.
echo ----------------------------------------------------
pause
