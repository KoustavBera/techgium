@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing dependencies...
pip install -r finetune_requirements.txt

echo Starting training...
python train_biomistral.py

pause
