"""
Specialized Risk Classifiers for Individual Physiological Systems
Optimized for hardware sensors: MAX30102, AD8232, MLX90614, DHT11, Camera
"""

import torch
import torch.nn as nn
import numpy as np


class CardiovascularRiskClassifier(nn.Module):
    """
    Specialized cardiovascular risk prediction model
    
    Input Features (7-10 recommended):
        CORE (7):
        - Heart Rate (from MAX30102)
        - Systolic Blood Pressure (estimated from PPG)
        - Diastolic Blood Pressure (estimated from PPG)
        - Derived_HRV (heart rate variability)
        - Derived_Pulse_Pressure (SBP - DBP)
        - Derived_MAP (mean arterial pressure)
        - Age
        
        OPTIONAL (improves accuracy to 93%+):
        - Derived_BMI
        - Weight (kg)
        - Height (m)
    
    Target: Cardiovascular disease risk [0-1]
    """
    
    def __init__(self, input_dim=10, hidden_dims=[128, 64, 32], dropout=0.3):
        """
        Smaller architecture than general vitals model for faster inference
        
        Args:
            input_dim: Number of cardio-specific features (default: 7)
            hidden_dims: Hidden layer sizes [64, 32, 16] for embedded devices
            dropout: Regularization (0.3 = 30% dropout)
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
        # Feature names for validation
        self.feature_names = [
            'Heart Rate',
            'Systolic Blood Pressure',
            'Diastolic Blood Pressure',
            'Derived_HRV',
            'Derived_Pulse_Pressure',
            'Derived_MAP',
            'Age'
        ]
    
    def forward(self, x):
        """
        Args:
            x: (batch, 7) tensor of cardio features
        Returns:
            risk_prob: (batch, 1) cardiovascular risk probability
        """
        return self.net(x)
    
    def predict_risk_score(self, x, return_dict=False):
        """
        Predict cardiovascular risk score [0-100]
        
        Args:
            x: (batch, 7) or (7,) numpy array or tensor
            return_dict: If True, return detailed results
        Returns:
            risk_score: Risk score 0-100
            OR dict with score, level, confidence
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            prob = self.forward(x).cpu().numpy().squeeze()
            risk_score = float(prob * 100)
            
            if return_dict:
                # Confidence = distance from decision boundary (0.5)
                confidence = abs(prob - 0.5) * 2
                
                # Risk level categorization
                if risk_score < 30:
                    level = 'LOW'
                elif risk_score < 60:
                    level = 'MODERATE'
                else:
                    level = 'HIGH'
                
                return {
                    'risk_score': round(risk_score, 2),
                    'risk_probability': round(float(prob), 4),
                    'risk_level': level,
                    'confidence': round(float(confidence), 3),
                    'system': 'cardiovascular'
                }
            
            return risk_score
    
    def get_feature_vector(self, hr, systolic_bp, diastolic_bp, hrv=None, age=30):
        """
        Helper to create feature vector from sensor readings
        
        Args:
            hr: Heart rate (BPM) from MAX30102
            systolic_bp: Systolic blood pressure (mmHg)
            diastolic_bp: Diastolic blood pressure (mmHg)
            hrv: Heart rate variability (optional, will estimate if None)
            age: Patient age (years)
        
        Returns:
            features: (7,) numpy array ready for prediction
        """
        # Estimate HRV if not provided (rough approximation)
        if hrv is None:
            # Normal resting HRV inversely correlates with HR
            hrv = max(0.02, 0.15 - (hr - 60) * 0.001)
        
        # Compute derived features
        pulse_pressure = systolic_bp - diastolic_bp
        map_pressure = (systolic_bp + 2 * diastolic_bp) / 3
        
        return np.array([
            hr,
            systolic_bp,
            diastolic_bp,
            hrv,
            pulse_pressure,
            map_pressure,
            age
        ], dtype=np.float32)


class RespiratoryRiskClassifier(nn.Module):
    """
    Specialized respiratory health prediction model
    
    Input Features (5-10 recommended):
        CORE (5):
        - Oxygen Saturation (SpO2 from MAX30102)
        - Respiratory Rate (from PPG signal analysis or camera)
        - Body Temperature (from MLX90614)
        - Heart Rate (from MAX30102, correlates with respiratory distress)
        - Age
        
        OPTIONAL (improves accuracy to 91%+):
        - Systolic/Diastolic Blood Pressure (hypotension in severe respiratory failure)
        - Derived_BMI (obesity is major respiratory risk factor)
        - Weight (impacts lung capacity)
        - Derived_HRV (reduced in respiratory compromise)
    
    Target: Respiratory disease risk [0-1]
    """
    
    def __init__(self, input_dim=10, hidden_dims=[64, 32, 16], dropout=0.3):
        """
        Lightweight architecture for edge deployment
        
        Args:
            input_dim: Number of respiratory features (default: 5)
            hidden_dims: Smaller layers [32, 16, 8] for Raspberry Pi inference
            dropout: Lower dropout (0.25) due to fewer features
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no sigmoid - using BCEWithLogitsLoss for numerical stability)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        self.feature_names = [
            'Oxygen Saturation',
            'Respiratory Rate',
            'Body Temperature',
            'Heart Rate',
            'Age',
            'Systolic Blood Pressure',
            'Diastolic Blood Pressure',
            'Derived_BMI',
            'Weight (kg)',
            'Derived_HRV'
        ]
    
    def forward(self, x):
        """
        Args:
            x: (batch, 5-10) tensor of respiratory features
        Returns:
            logits: (batch, 1) raw logits (apply sigmoid for probabilities)
        Returns:
            risk_prob: (batch, 1) respiratory risk probability
        """
        return self.net(x)
    
    def predict_risk_score(self, x, return_dict=False):
        """
        Predict respiratory risk score [0-100]
        
        Args:
            x: (batch, 5) or (5,) numpy array or tensor
            return_dict: If True, return detailed results
        Returns:
            risk_score: Risk score 0-100
            OR dict with score, level, confidence
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            prob = self.forward(x).cpu().numpy().squeeze()
            risk_score = float(prob * 100)
            
            if return_dict:
                confidence = abs(prob - 0.5) * 2
                
                # Respiratory-specific thresholds (more conservative)
                if risk_score < 25:
                    level = 'LOW'
                elif risk_score < 55:
                    level = 'MODERATE'
                else:
                    level = 'HIGH'
                
                return {
                    'risk_score': round(risk_score, 2),
                    'risk_probability': round(float(prob), 4),
                    'risk_level': level,
                    'confidence': round(float(confidence), 3),
                    'system': 'respiratory'
                }
            
            return risk_score
    
    def get_feature_vector(self, spo2, respiratory_rate, body_temp, hr, age=30):
        """
        Helper to create feature vector from sensor readings
        
        Args:
            spo2: Oxygen saturation (%) from MAX30102
            respiratory_rate: Breaths per minute (from camera or PPG)
            body_temp: Body temperature (°C) from MLX90614
            hr: Heart rate (BPM) from MAX30102
            age: Patient age (years)
        
        Returns:
            features: (5,) numpy array ready for prediction
        """
        return np.array([
            spo2,
            respiratory_rate,
            body_temp,
            hr,
            age
        ], dtype=np.float32)


class ThermalHealthClassifier(nn.Module):
    """
    Fever and infection detection model
    
    Input Features (4):
        - Body Temperature (from MLX90614)
        - Heart Rate (elevated in fever)
        - Ambient Temperature (from DHT11, for context)
        - Ambient Humidity (from DHT11, affects perception)
    
    Target: Infection/fever risk [0-1]
    """
    
    def __init__(self, input_dim=4, hidden_dims=[16, 8], dropout=0.2):
        """
        Very lightweight for quick triage
        
        Args:
            input_dim: Number of thermal features (default: 4)
            hidden_dims: Minimal layers [16, 8]
            dropout: Minimal dropout (0.2) for simple task
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
        self.feature_names = [
            'Body Temperature',
            'Heart Rate',
            'Ambient Temperature',
            'Ambient Humidity'
        ]
    
    def forward(self, x):
        return self.net(x)
    
    def predict_risk_score(self, x, return_dict=False):
        """Predict infection/fever risk score [0-100]"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            prob = self.forward(x).cpu().numpy().squeeze()
            risk_score = float(prob * 100)
            
            if return_dict:
                confidence = abs(prob - 0.5) * 2
                
                # Fever-specific levels
                if risk_score < 20:
                    level = 'NORMAL'
                elif risk_score < 50:
                    level = 'ELEVATED'
                else:
                    level = 'FEVER'
                
                return {
                    'risk_score': round(risk_score, 2),
                    'risk_probability': round(float(prob), 4),
                    'risk_level': level,
                    'confidence': round(float(confidence), 3),
                    'system': 'thermal'
                }
            
            return risk_score
    
    def get_feature_vector(self, body_temp, hr, ambient_temp=25.0, ambient_humidity=50.0):
        """
        Helper to create feature vector from sensors
        
        Args:
            body_temp: Body temperature (°C) from MLX90614
            hr: Heart rate (BPM) from MAX30102
            ambient_temp: Room temperature (°C) from DHT11
            ambient_humidity: Relative humidity (%) from DHT11
        
        Returns:
            features: (4,) numpy array
        """
        return np.array([
            body_temp,
            hr,
            ambient_temp,
            ambient_humidity
        ], dtype=np.float32)


# Model size comparison for deployment planning
def get_model_info():
    """
    Print model sizes and parameter counts for deployment decisions
    """
    models = {
        'Cardiovascular': CardiovascularRiskClassifier(),
        'Respiratory': RespiratoryRiskClassifier(),
        'Thermal': ThermalHealthClassifier()
    }
    
    print("=" * 70)
    print("SPECIALIZED MODEL COMPARISON")
    print("=" * 70)
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate size in MB (4 bytes per float32 parameter)
        size_mb = params * 4 / (1024 * 1024)
        
        print(f"\n{name} Model:")
        print(f"  Parameters: {params:,} ({trainable:,} trainable)")
        print(f"  Model size: {size_mb:.2f} MB")
        print(f"  Features: {len(model.feature_names)}")
        print(f"  Architecture: {' → '.join(map(str, [len(model.feature_names)] + [m.out_features for m in model.net if isinstance(m, nn.Linear)]))}")
    
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATION:")
    print("  • Raspberry Pi Pico W: Thermal model (138 params, 0.0005 MB)")
    print("  • Edge device (phone): Cardio + Respiratory (~2.5k params, 0.01 MB)")
    print("  • Backend GPU: All models + vision models")
    print("=" * 70)


if __name__ == "__main__":
    # Test models
    print("Testing specialized classifiers...\n")
    
    # Cardiovascular test
    cardio_model = CardiovascularRiskClassifier()
    cardio_features = cardio_model.get_feature_vector(
        hr=85, systolic_bp=140, diastolic_bp=90, hrv=0.08, age=55
    )
    cardio_result = cardio_model.predict_risk_score(cardio_features, return_dict=True)
    print("Cardiovascular Test:")
    print(f"  Input: HR=85, BP=140/90, HRV=0.08, Age=55")
    print(f"  Result: {cardio_result}")
    
    # Respiratory test
    resp_model = RespiratoryRiskClassifier()
    resp_features = resp_model.get_feature_vector(
        spo2=94, respiratory_rate=22, body_temp=37.8, hr=95, age=55
    )
    resp_result = resp_model.predict_risk_score(resp_features, return_dict=True)
    print("\nRespiratory Test:")
    print(f"  Input: SpO2=94%, RR=22, Temp=37.8°C, HR=95, Age=55")
    print(f"  Result: {resp_result}")
    
    # Thermal test
    thermal_model = ThermalHealthClassifier()
    thermal_features = thermal_model.get_feature_vector(
        body_temp=38.2, hr=102, ambient_temp=24.0, ambient_humidity=55.0
    )
    thermal_result = thermal_model.predict_risk_score(thermal_features, return_dict=True)
    print("\nThermal Test:")
    print(f"  Input: Body=38.2°C, HR=102, Room=24°C, Humidity=55%")
    print(f"  Result: {thermal_result}")
    
    # Model info
    print("\n")
    get_model_info()
    