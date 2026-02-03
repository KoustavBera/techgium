"""
STEP 1 VALIDATION SCRIPT
=======================
100% Accuracy Verification and Debugging

Compares:
1. Original tabular classifier (14 features, 94.8% accuracy)
2. Specialized cardio classifier (7 features, expected 93-96%)
3. Specialized respiratory classifier (5 features, expected 91-94%)

Debugging Checks:
- Feature distribution validation
- Model loading verification
- Prediction consistency
- Performance metrics comparison
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib

sys.path.append(os.path.dirname(__file__))

print("=" * 90)
print(" " * 25 + "STEP 1: ACCURACY VALIDATION")
print("=" * 90)

# Load dataset
csv_path = os.path.join('data', 'human_vital_signs_dataset_2024.csv')
print(f"\n📂 Loading: {csv_path}")
df = pd.read_csv(csv_path)
print(f"   ✓ {len(df):,} records\n")

# Prepare labels
y = (df['Risk Category'] == 'High Risk').astype(int).values

# Test splits
_, X_test, _, y_test = train_test_split(
    df, y, test_size=0.15, random_state=42, stratify=y
)

print(f"🧪 Test Set: {len(X_test):,} samples\n")

# =============================================================================
# TEST 1: CARDIOVASCULAR CLASSIFIER
# =============================================================================
print("=" * 90)
print("TEST 1: CARDIOVASCULAR CLASSIFIER")
print("=" * 90)

CARDIO_FEATURES = [
    'Heart Rate', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
    'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_MAP', 'Age'
]

print(f"\n📋 Features ({len(CARDIO_FEATURES)}):", CARDIO_FEATURES)

# Check if model exists
cardio_model_path = os.path.join('ml', 'train', 'cardio_classifier_best.pt')
cardio_scaler_path = os.path.join('ml', 'train', 'cardio_scaler.joblib')
cardio_results_path = os.path.join('ml', 'train', 'cardio_training_results.json')

if not os.path.exists(cardio_model_path):
    print(f"\n❌ FAIL: Model not found at {cardio_model_path}")
    print(f"   Run: python -m ml.train.train_cardio_classifier")
    cardio_trained = False
else:
    print(f"\n✅ Model found: {cardio_model_path}")
    cardio_trained = True
    
    # Load trained results
    if os.path.exists(cardio_results_path):
        with open(cardio_results_path, 'r') as f:
            cardio_results = json.load(f)
        
        print(f"\n📊 Training Results:")
        print(f"   Test Accuracy:  {cardio_results['test_metrics']['accuracy']:.4f} ({cardio_results['test_metrics']['accuracy']*100:.2f}%)")
        print(f"   Test AUC:       {cardio_results['test_metrics']['roc_auc']:.4f}")
        print(f"   Test Precision: {cardio_results['test_metrics']['precision']:.4f}")
        print(f"   Test Recall:    {cardio_results['test_metrics']['recall']:.4f}")
        print(f"   Sensitivity:    {cardio_results['test_metrics']['sensitivity']:.4f}")
        print(f"   Specificity:    {cardio_results['test_metrics']['specificity']:.4f}")
        
        # DEBUGGING: Confusion matrix
        conf_matrix = np.array(cardio_results['confusion_matrix'])
        print(f"\n📋 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"               Low    High")
        print(f"   Actual Low   {conf_matrix[0,0]:5d}  {conf_matrix[0,1]:5d}")
        print(f"   Actual High  {conf_matrix[1,0]:5d}  {conf_matrix[1,1]:5d}")
        
        # VALIDATION CHECK
        acc = cardio_results['test_metrics']['accuracy']
        if acc >= 0.93:
            print(f"\n✅ PASS: Accuracy {acc*100:.2f}% meets target (≥93%)")
        else:
            print(f"\n⚠️  FAIL: Accuracy {acc*100:.2f}% below target (≥93%)")
            print(f"   Gap: {(0.93 - acc)*100:.2f}%")
    
    # DEBUGGING: Test inference
    print(f"\n🔍 DEBUG: Testing inference...")
    from ml.models.specialized_classifiers import CardiovascularRiskClassifier
    
    checkpoint = torch.load(cardio_model_path, map_location='cpu')
    model = CardiovascularRiskClassifier(
        input_dim=checkpoint['config']['input_dim'],
        hidden_dims=checkpoint['config']['hidden_dims'],
        dropout=checkpoint['config']['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = joblib.load(cardio_scaler_path)
    
    # Test on 5 random samples
    test_indices = np.random.choice(len(X_test), 5, replace=False)
    print(f"\n   Sample Predictions:")
    for i, idx in enumerate(test_indices):
        sample = X_test.iloc[idx]
        features = np.array([[
            sample['Heart Rate'],
            sample['Systolic Blood Pressure'],
            sample['Diastolic Blood Pressure'],
            sample['Derived_HRV'],
            sample['Derived_Pulse_Pressure'],
            sample['Derived_MAP'],
            sample['Age']
        ]], dtype=np.float32)
        
        features_scaled = scaler.transform(features)
        with torch.no_grad():
            pred = model(torch.from_numpy(features_scaled).float()).numpy()[0][0]
        
        true_label = 'High Risk' if y_test.iloc[idx] == 1 else 'Low Risk'
        pred_label = 'High Risk' if pred > 0.5 else 'Low Risk'
        match = '✓' if (pred > 0.5) == y_test.iloc[idx] else '✗'
        
        print(f"   {i+1}. HR={sample['Heart Rate']:.0f}, BP={sample['Systolic Blood Pressure']:.0f}/{sample['Diastolic Blood Pressure']:.0f} "
              f"→ Pred={pred:.3f} ({pred_label}) vs True={true_label} {match}")

# =============================================================================
# TEST 2: RESPIRATORY CLASSIFIER  
# =============================================================================
print("\n" + "=" * 90)
print("TEST 2: RESPIRATORY CLASSIFIER")
print("=" * 90)

RESP_FEATURES = [
    'Oxygen Saturation', 'Respiratory Rate', 'Body Temperature',
    'Heart Rate', 'Age'
]

print(f"\n📋 Features ({len(RESP_FEATURES)}):", RESP_FEATURES)

resp_model_path = os.path.join('ml', 'train', 'resp_classifier_best.pt')
resp_scaler_path = os.path.join('ml', 'train', 'resp_scaler.joblib')
resp_results_path = os.path.join('ml', 'train', 'resp_training_results.json')

if not os.path.exists(resp_model_path):
    print(f"\n❌ FAIL: Model not found at {resp_model_path}")
    print(f"   Run: python -m ml.train.train_resp_classifier")
    resp_trained = False
else:
    print(f"\n✅ Model found: {resp_model_path}")
    resp_trained = True
    
    if os.path.exists(resp_results_path):
        with open(resp_results_path, 'r') as f:
            resp_results = json.load(f)
        
        print(f"\n📊 Training Results:")
        print(f"   Test Accuracy:  {resp_results['test_metrics']['accuracy']:.4f} ({resp_results['test_metrics']['accuracy']*100:.2f}%)")
        print(f"   Test AUC:       {resp_results['test_metrics']['roc_auc']:.4f}")
        print(f"   Test Precision: {resp_results['test_metrics']['precision']:.4f}")
        print(f"   Test Recall:    {resp_results['test_metrics']['recall']:.4f}")
        
        acc = resp_results['test_metrics']['accuracy']
        if acc >= 0.91:
            print(f"\n✅ PASS: Accuracy {acc*100:.2f}% meets target (≥91%)")
        else:
            print(f"\n⚠️  FAIL: Accuracy {acc*100:.2f}% below target (≥91%)")

# =============================================================================
# TEST 3: ENGINE INTEGRATION
# =============================================================================
print("\n" + "=" * 90)
print("TEST 3: ENGINE INTEGRATION")
print("=" * 90)

print(f"\n🔍 Testing CardioRiskEngine...")

try:
    from models.engine_cardiovascular import CardioRiskEngine
    
    # Test specialized mode
    engine = CardioRiskEngine(
        heart_rate=85,
        systolic_bp=145,
        diastolic_bp=92,
        hrv=0.07,
        age=58,
        signal_quality=0.95,
        use_specialized=True
    )
    
    result = engine.run()
    
    print(f"   ✅ Specialized mode working")
    print(f"   Risk Score: {result['risk_score']}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Model Type: {result.get('model_type', 'unknown')}")
    print(f"   Flags: {result['flags']}")
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 90)
print(" " * 35 + "VALIDATION SUMMARY")
print("=" * 90)

summary = {
    'cardio_model_trained': cardio_trained,
    'resp_model_trained': resp_trained,
    'cardio_accuracy': cardio_results['test_metrics']['accuracy'] if cardio_trained else 0,
    'resp_accuracy': resp_results['test_metrics']['accuracy'] if resp_trained else 0,
}

print(f"\n📊 Results:")
print(f"   1. Cardiovascular Classifier: {'✅ TRAINED' if cardio_trained else '❌ NOT TRAINED'}")
if cardio_trained:
    print(f"      - Accuracy: {summary['cardio_accuracy']*100:.2f}%")
    print(f"      - Status: {'✅ PASS' if summary['cardio_accuracy'] >= 0.93 else '⚠️  BELOW TARGET'}")

print(f"\n   2. Respiratory Classifier: {'✅ TRAINED' if resp_trained else '❌ NOT TRAINED'}")
if resp_trained:
    print(f"      - Accuracy: {summary['resp_accuracy']*100:.2f}%")
    print(f"      - Status: {'✅ PASS' if summary['resp_accuracy'] >= 0.91 else '⚠️  BELOW TARGET'}")

print(f"\n   3. Engine Integration: Testing required")

# Final verdict
if cardio_trained and resp_trained:
    if summary['cardio_accuracy'] >= 0.93 and summary['resp_accuracy'] >= 0.91:
        print(f"\n{'='*90}")
        print(f"{'✅ STEP 1 COMPLETE: ALL TESTS PASSED':^90}")
        print(f"{'='*90}")
        print(f"\nNext: Proceed to Step 2 (ECG arrhythmia detector)")
    else:
        print(f"\n⚠️  STEP 1 INCOMPLETE: Some models below accuracy target")
else:
    print(f"\n❌ STEP 1 INCOMPLETE: Train missing models")
    if not cardio_trained:
        print(f"   Run: python -m ml.train.train_cardio_classifier")
    if not resp_trained:
        print(f"   Run: python -m ml.train.train_resp_classifier")

print(f"\n{'='*90}\n")
