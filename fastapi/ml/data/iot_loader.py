"""
Healthcare IoT Dataset Loader
Loads IoT sensor data with multi-sensor fusion and patient-level splits
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from .cyclical_encoding import add_cyclical_time


class IoTDataLoader:
    """
    Loads and preprocesses Healthcare IoT Dataset (202 samples)
    Implements patient-level splits and multi-sensor pivoting
    """
    
    def __init__(self, csv_path=None, window_size=256, test_size=0.3, random_state=42):
        """
        Args:
            csv_path: Path to healthcare_iot_target_dataset.csv
            window_size: Number of timesteps per window (default: 256)
            test_size: Fraction of patients for test set (default: 0.3, larger due to small dataset)
            random_state: Random seed for reproducibility
        """
        if csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            csv_path = os.path.join(base_dir, 'data', 'healthcare_iot_target_dataset.csv')
        
        self.csv_path = csv_path
        self.window_size = window_size
        self.test_size = test_size
        self.random_state = random_state
        
        self.df = None
        self.train_patients = None
        self.test_patients = None
        
    def load_data(self):
        """Load CSV and preprocess"""
        print(f"📂 Loading IoT dataset from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        
        # Fix temperature data artifact (some rows have wrong values)
        self.df['Temperature (°C)'] = pd.to_numeric(self.df['Temperature (°C)'], errors='coerce')
        self.df['Temperature (°C)'].fillna(self.df['Temperature (°C)'].mean(), inplace=True)
        
        print(f"   ✓ Loaded {len(self.df):,} records")
        print(f"   ✓ Unique patients: {self.df['Patient_ID'].nunique()}")
        print(f"   ✓ Sensor types: {self.df['Sensor_Type'].unique()}")
        print(f"   ✓ Date range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
        
        return self.df
    
    def create_patient_splits(self):
        """Split patients into train/test (no val due to small size)"""
        if self.df is None:
            self.load_data()
        
        unique_patients = self.df['Patient_ID'].unique()
        n_patients = len(unique_patients)
        
        train_patients, test_patients = train_test_split(
            unique_patients,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        self.train_patients = train_patients
        self.test_patients = test_patients
        
        print(f"\n📊 Patient-level splits:")
        print(f"   Train: {len(train_patients)} patients ({len(train_patients)/n_patients*100:.1f}%)")
        print(f"   Test:  {len(test_patients)} patients ({len(test_patients)/n_patients*100:.1f}%)")
        
        return train_patients, test_patients
    
    def pivot_multi_sensor(self, patient_df):
        """
        Pivot sensor readings into wide format for multi-sensor fusion
        
        Args:
            patient_df: DataFrame for single patient with multiple sensor readings
        
        Returns:
            DataFrame with columns for each sensor's measurements
        """
        # Group by timestamp and aggregate sensor readings
        pivot_data = []
        
        for timestamp, group in patient_df.groupby('Timestamp'):
            row = {'Timestamp': timestamp}
            
            # Extract measurements from each sensor type
            for sensor_type in group['Sensor_Type'].unique():
                sensor_data = group[group['Sensor_Type'] == sensor_type].iloc[0]
                row[f'{sensor_type}_Temp'] = sensor_data['Temperature (°C)']
                row[f'{sensor_type}_HR'] = sensor_data['Heart_Rate (bpm)']
                row[f'{sensor_type}_SysBP'] = sensor_data['Systolic_BP (mmHg)']
                row[f'{sensor_type}_DiaBP'] = sensor_data['Diastolic_BP (mmHg)']
                row[f'{sensor_type}_Battery'] = sensor_data['Battery_Level (%)']
            
            # Add target labels
            row['Target_Health_Status'] = group['Target_Health_Status'].iloc[0]
            pivot_data.append(row)
        
        pivot_df = pd.DataFrame(pivot_data)
        pivot_df.sort_values('Timestamp', inplace=True)
        
        return pivot_df
    
    def create_patient_timeseries(self, patient_df, target_length=256):
        """
        Create time series for patient with interpolation to target length
        
        Args:
            patient_df: DataFrame for single patient
            target_length: Target number of timesteps (default: 256)
        
        Returns:
            X: numpy array (2, target_length) with HR and SpO2-approximated values
            y: health status label (0=Unhealthy, 1=Healthy)
            timestamps: array of timestamps
        """
        patient_df = patient_df.sort_values('Timestamp').copy()
        
        # Extract HR as primary signal
        hr_values = patient_df['Heart_Rate (bpm)'].values
        
        # Use Temperature as proxy for SpO2 (scaled to SpO2 range)
        # This is a limitation of the dataset - adjust based on your needs
        temp_values = patient_df['Temperature (°C)'].values
        # Scale temperature 36-38°C to approximate SpO2 range 94-100%
        spo2_proxy = 94 + (temp_values - 36) * 3  # Rough approximation
        spo2_proxy = np.clip(spo2_proxy, 94, 100)
        
        timestamps = patient_df['Timestamp'].values
        
        # Interpolate to target length if needed
        if len(hr_values) < target_length:
            # Upsample using linear interpolation
            from scipy.interpolate import interp1d
            
            x_old = np.linspace(0, 1, len(hr_values))
            x_new = np.linspace(0, 1, target_length)
            
            hr_interp = interp1d(x_old, hr_values, kind='linear')
            spo2_interp = interp1d(x_old, spo2_proxy, kind='linear')
            
            hr_values = hr_interp(x_new)
            spo2_proxy = spo2_interp(x_new)
            
            # Interpolate timestamps
            timestamps_numeric = patient_df['Timestamp'].astype(np.int64).values
            ts_interp = interp1d(x_old, timestamps_numeric, kind='linear')
            timestamps = ts_interp(x_new).astype('datetime64[ns]')
        
        elif len(hr_values) > target_length:
            # Downsample by taking evenly spaced points
            indices = np.linspace(0, len(hr_values) - 1, target_length, dtype=int)
            hr_values = hr_values[indices]
            spo2_proxy = spo2_proxy[indices]
            timestamps = timestamps[indices]
        
        # Stack as (2, target_length)
        X = np.stack([hr_values, spo2_proxy], axis=0)
        
        # Extract health status label
        health_status = patient_df['Target_Health_Status'].iloc[0]
        y = 1 if health_status == 'Healthy' else 0
        
        return X, y, timestamps
    
    def get_dataset(self, split='train', include_time_features=True, normalize=True):
        """
        Get processed dataset for specified split
        
        Args:
            split: 'train' or 'test'
            include_time_features: Add cyclical time encoding
            normalize: Apply z-score normalization
        
        Returns:
            X: numpy array of shape (N, channels, window_size)
            y: numpy array of health status labels
            metadata: dict with patient IDs
        """
        if self.train_patients is None:
            self.create_patient_splits()
        
        if split == 'train':
            patient_ids = self.train_patients
        elif split == 'test':
            patient_ids = self.test_patients
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
        
        split_df = self.df[self.df['Patient_ID'].isin(patient_ids)].copy()
        
        print(f"\n🔧 Processing IoT {split} split...")
        print(f"   Patients: {len(patient_ids)}")
        
        all_windows = []
        all_labels = []
        all_timestamps = []
        patient_id_map = []
        
        for patient_id in patient_ids:
            patient_df = split_df[split_df['Patient_ID'] == patient_id]
            
            if len(patient_df) < 3:
                continue
            
            try:
                X, y, timestamps = self.create_patient_timeseries(patient_df, self.window_size)
                all_windows.append(X)
                all_labels.append(y)
                all_timestamps.append(timestamps)
                patient_id_map.append(patient_id)
            except Exception as e:
                print(f"   ⚠️  Skipping patient {patient_id}: {e}")
                continue
        
        # Stack all windows
        X = np.array(all_windows) if all_windows else np.array([])  # Shape: (N, 2, window_size)
        y = np.array(all_labels)
        
        print(f"   ✓ Generated {len(X)} samples")
        
        # Add time features
        if include_time_features and len(X) > 0:
            first_timestamps = [ts[0] for ts in all_timestamps]
            X = add_cyclical_time(X, np.array(first_timestamps))
            print(f"   ✓ Added cyclical time features")
        
        # Normalize
        if normalize and len(X) > 0:
            for ch in range(2):
                mean = X[:, ch, :].mean()
                std = X[:, ch, :].std()
                X[:, ch, :] = (X[:, ch, :] - mean) / (std + 1e-6)
            print(f"   ✓ Normalized channels")
        
        # Class distribution
        if len(y) > 0:
            n_unhealthy = (y == 0).sum()
            n_healthy = (y == 1).sum()
            print(f"   ✓ Health status: {n_unhealthy} Unhealthy ({n_unhealthy/len(y)*100:.1f}%), {n_healthy} Healthy ({n_healthy/len(y)*100:.1f}%)")
        
        metadata = {
            'patient_ids': patient_id_map,
            'timestamps': all_timestamps,
            'split': split
        }
        
        return X, y, metadata


def load_iot_dataset(csv_path=None, window_size=256, include_time_features=True):
    """Convenience function to load train and test splits"""
    loader = IoTDataLoader(csv_path=csv_path, window_size=window_size)
    loader.load_data()
    
    datasets = {}
    for split in ['train', 'test']:
        X, y, metadata = loader.get_dataset(split, include_time_features=include_time_features)
        datasets[split] = (X, y, metadata)
    
    return datasets
