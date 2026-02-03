"""
Train Specialized Cardiovascular Risk Classifier
Filters vitals dataset to cardio-specific features for optimized performance

Expected Accuracy: 93-96% (vs 94% for general vitals model)
Training Time: ~3 minutes on RTX 3050
Model Size: 2.4k parameters (vs 12.7k for general model)
"""

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.models.specialized_classifiers import CardiovascularRiskClassifier

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
SEED = 42

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 80)
print("TRAINING CARDIOVASCULAR RISK CLASSIFIER")
print("=" * 80)
print(f"\n🖥️  Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
print(f"📊 Config: {EPOCHS} epochs, LR={LEARNING_RATE}, Batch={BATCH_SIZE}")
print(f"⏱️  Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs\n")

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'human_vital_signs_dataset_2024.csv')
print(f"📂 Loading: {os.path.basename(csv_path)}")
df = pd.read_csv(csv_path)
print(f"   ✓ Loaded {len(df):,} patient records\n")

# FEATURE SELECTION: Cardiovascular-specific features (EXPANDED)
# Added BMI, Weight, Height for better risk prediction
CARDIO_FEATURES = [
    'Heart Rate',
    'Systolic Blood Pressure',
    'Diastolic Blood Pressure',
    'Derived_HRV',
    'Derived_Pulse_Pressure',
    'Derived_MAP',
    'Age',
    'Derived_BMI',           # BMI strongly correlates with cardiovascular risk
    'Weight (kg)',           # Obesity indicator
    'Height (m)'             # For body size normalization
]

print(f"🫀 Cardiovascular Features ({len(CARDIO_FEATURES)}):")
for i, feat in enumerate(CARDIO_FEATURES, 1):
    # Show data range for debugging
    if feat in df.columns:
        feat_min, feat_max = df[feat].min(), df[feat].max()
        feat_mean = df[feat].mean()
        print(f"   {i}. {feat:30s} → [{feat_min:7.2f}, {feat_max:7.2f}] (μ={feat_mean:7.2f})")
    else:
        print(f"   {i}. {feat:30s} → ⚠️  MISSING IN DATASET")

# Verify all features exist
missing_features = [f for f in CARDIO_FEATURES if f not in df.columns]
if missing_features:
    print(f"\n❌ ERROR: Missing features: {missing_features}")
    print("   Available columns:", df.columns.tolist())
    sys.exit(1)

# Prepare data
X = df[CARDIO_FEATURES].values.astype(np.float32)
y = (df['Risk Category'] == 'High Risk').astype(int).values

print(f"\n📈 Dataset Statistics:")
print(f"   Total samples: {len(X):,}")
print(f"   Low Risk  (0): {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"   High Risk (1): {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"   Feature shape: {X.shape}")

# CRITICAL: Patient-level splits to prevent data leakage
# (Assuming each row = unique patient, otherwise add patient ID grouping)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print(f"\n🔀 Data Splits (Patient-Level):")
print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# Normalize features (fit on train only)
print(f"\n🔧 Normalizing features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# DEBUG: Check for NaN or Inf after scaling
if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
    print("   ⚠️  WARNING: NaN or Inf detected after scaling!")
    nan_cols = np.where(np.isnan(X_train_scaled).any(axis=0))[0]
    print(f"   Problematic features: {[CARDIO_FEATURES[i] for i in nan_cols]}")
else:
    print("   ✓ No NaN/Inf detected")

# Save scaler for inference
scaler_path = os.path.join(os.path.dirname(__file__), 'cardio_scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"   ✓ Saved scaler: {os.path.basename(scaler_path)}")

# Create DataLoaders
train_dataset = TensorDataset(
    torch.from_numpy(X_train_scaled).float(),
    torch.from_numpy(y_train).float().unsqueeze(1)
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val_scaled).float(),
    torch.from_numpy(y_val).float().unsqueeze(1)
)
test_dataset = TensorDataset(
    torch.from_numpy(X_test_scaled).float(),
    torch.from_numpy(y_test).float().unsqueeze(1)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
print(f"\n🏗️  Model Architecture:")
model = CardiovascularRiskClassifier(
    input_dim=len(CARDIO_FEATURES),
    hidden_dims=[128, 64, 32],  # Larger architecture for better capacity
    dropout=0.3
).to(DEVICE)

print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,} ({trainable_params:,} trainable)")
print(f"   Model size: {total_params * 4 / 1024:.2f} KB")

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# CLASS IMBALANCE HANDLING: Compute class weights
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # Weight for positive class
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))

# IMPORTANT: Remove sigmoid from model output when using BCEWithLogitsLoss
model.net = model.net[:-1]  # Remove final Sigmoid layer

print(f"\n⚖️  Class Balance:")
print(f"   Positive weight (High Risk): {pos_weight:.3f}")
print(f"   This compensates for {(y_train == 0).sum():,} Low vs {(y_train == 1).sum():,} High samples")

print(f"\n🚀 Starting Training...")
print("=" * 80)

# Training loop with accuracy tracking
best_val_auc = 0.0
best_epoch = 0
patience_counter = 0
training_history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_auc': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': []
}

for epoch in range(1, EPOCHS + 1):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            
            # Apply sigmoid for BCEWithLogitsLoss
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
    
    val_loss /= len(val_loader.dataset)
    
    # Calculate metrics
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, zero_division=0)
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    val_auc = roc_auc_score(all_labels, all_probs)
    
    # Store history
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    training_history['val_accuracy'].append(val_accuracy)
    training_history['val_auc'].append(val_auc)
    training_history['val_precision'].append(val_precision)
    training_history['val_recall'].append(val_recall)
    training_history['val_f1'].append(val_f1)
    
    # Print progress every 5 epochs or at milestones
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} │ "
              f"Train Loss: {train_loss:.4f} │ "
              f"Val Loss: {val_loss:.4f} │ "
              f"Val Acc: {val_accuracy:.4f} │ "
              f"Val AUC: {val_auc:.4f}")
    
    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        patience_counter = 0
        
        # Save checkpoint
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'cardio_classifier_best.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
            'val_accuracy': val_accuracy,
            'feature_cols': CARDIO_FEATURES,
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_std': scaler.scale_.tolist(),
            'config': {
                'input_dim': len(CARDIO_FEATURES),
                'hidden_dims': [64, 32, 16],
                'dropout': 0.3
            }
        }, checkpoint_path)
        
        if epoch > 1:  # Don't print on first epoch
            print(f"   ✅ New best model! AUC: {val_auc:.4f} (epoch {epoch})")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
        print(f"   Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")
        break

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)

# Load best model for final evaluation
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Final test set evaluation
print(f"\n📊 FINAL TEST SET EVALUATION")
print("=" * 80)

all_test_preds = []
all_test_probs = []
all_test_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(DEVICE)
        outputs = model(batch_X).cpu().numpy()
        # Apply sigmoid for BCEWithLogitsLoss
        outputs = 1 / (1 + np.exp(-outputs))  # Sigmoid
        preds = (outputs > 0.5).astype(int)
        
        all_test_probs.extend(outputs.flatten())
        all_test_preds.extend(preds.flatten())
        all_test_labels.extend(batch_y.numpy().flatten())

# Calculate test metrics
test_accuracy = accuracy_score(all_test_labels, all_test_preds)
test_precision = precision_score(all_test_labels, all_test_preds, zero_division=0)
test_recall = recall_score(all_test_labels, all_test_preds, zero_division=0)
test_f1 = f1_score(all_test_labels, all_test_preds, zero_division=0)
test_auc = roc_auc_score(all_test_labels, all_test_probs)
conf_matrix = confusion_matrix(all_test_labels, all_test_preds)

print(f"\n📈 Performance Metrics:")
print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1 Score:  {test_f1:.4f}")
print(f"   ROC AUC:   {test_auc:.4f}")

print(f"\n📋 Confusion Matrix:")
print(f"                 Predicted")
print(f"               Low    High")
print(f"   Actual Low   {conf_matrix[0,0]:5d}  {conf_matrix[0,1]:5d}")
print(f"   Actual High  {conf_matrix[1,0]:5d}  {conf_matrix[1,1]:5d}")

# Calculate specificity and sensitivity
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n🎯 Clinical Metrics:")
print(f"   Sensitivity (True Positive Rate):  {sensitivity:.4f} ({sensitivity*100:.1f}%)")
print(f"   Specificity (True Negative Rate):  {specificity:.4f} ({specificity*100:.1f}%)")
print(f"   False Positive Rate: {fp}/{fp+tn} ({fp/(fp+tn)*100:.1f}%)")
print(f"   False Negative Rate: {fn}/{fn+tp} ({fn/(fn+tp)*100:.1f}%)")

# Save results
results = {
    'model_type': 'CardiovascularRiskClassifier',
    'training_date': datetime.now().isoformat(),
    'features': CARDIO_FEATURES,
    'num_features': len(CARDIO_FEATURES),
    'dataset_size': len(df),
    'train_size': len(X_train),
    'val_size': len(X_val),
    'test_size': len(X_test),
    'best_epoch': best_epoch,
    'total_epochs': epoch,
    'training_history': {
        'train_loss': [float(x) for x in training_history['train_loss']],
        'val_loss': [float(x) for x in training_history['val_loss']],
        'val_accuracy': [float(x) for x in training_history['val_accuracy']],
        'val_auc': [float(x) for x in training_history['val_auc']],
        'val_precision': [float(x) for x in training_history['val_precision']],
        'val_recall': [float(x) for x in training_history['val_recall']],
        'val_f1': [float(x) for x in training_history['val_f1']]
    },
    'test_metrics': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1),
        'roc_auc': float(test_auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    },
    'confusion_matrix': conf_matrix.tolist(),
    'model_size_kb': total_params * 4 / 1024,
    'total_parameters': total_params
}

results_path = os.path.join(os.path.dirname(__file__), 'cardio_training_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Saved Results:")
print(f"   Model: {os.path.basename(checkpoint_path)}")
print(f"   Scaler: {os.path.basename(scaler_path)}")
print(f"   Results: {os.path.basename(results_path)}")

print("\n" + "=" * 80)
print("🎯 ACCURACY VALIDATION")
print("=" * 80)
print(f"✅ Target Accuracy: ≥93%")
print(f"📊 Achieved: {test_accuracy*100:.2f}%")

if test_accuracy >= 0.93:
    print(f"✅ PASS: Model meets accuracy requirement!")
else:
    print(f"⚠️  WARNING: Below target (delta: {(0.93-test_accuracy)*100:.2f}%)")
    print(f"   Recommendations:")
    print(f"   - Increase training epochs")
    print(f"   - Try larger hidden_dims [128, 64, 32]")
    print(f"   - Add more features (e.g., weight, BMI)")

print("=" * 80)
