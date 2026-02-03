"""
Human Vital Signs Dataset Loader
Loads real patient data with SpO2/HR time-series extraction and patient-level splits
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from .cyclical_encoding import add_cyclical_time


class VitalsDataLoader:
    """
    Loads and preprocesses Human Vital Signs Dataset (200k samples)
    Implements patient-level holdout splits to prevent data leakage
    """
    
    def __init__(self, csv_path=None, window_size=256, stride=128, test_size=0.15, val_size=0.15, random_state=42):
        """
        Args:
            csv_path: Path to human_vital_signs_dataset_2024.csv
            window_size: Number of timesteps per window (default: 256)
            stride: Step size for sliding window (default: 128 = 50% overlap)
            test_size: Fraction of patients for test set (default: 0.15)
            val_size: Fraction of training patients for validation (default: 0.15)
            random_state: Random seed for reproducibility
        """
        if csv_path is None:
            # Default path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            csv_path = os.path.join(base_dir, 'data', 'human_vital_signs_dataset_2024.csv')
        
        self.csv_path = csv_path
        self.window_size = window_size
        self.stride = stride
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        self.df = None
        self.train_patients = None
        self.val_patients = None
        self.test_patients = None
        
    def load_data(self):
        """Load CSV and parse timestamps"""
        print(f"📂 Loading vitals dataset from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        
        print(f"   ✓ Loaded {len(self.df):,} records")
        print(f"   ✓ Columns: {list(self.df.columns)}")
        print(f"   ✓ Date range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
        
        # Check for unique patients
        n_patients = self.df['Patient ID'].nunique()
        print(f"   ✓ Unique patients: {n_patients:,}")
        
        return self.df
    
    def create_patient_splits(self):
        """
        Split patients (not samples) into train/val/test to prevent leakage
        Ensures no patient appears in multiple splits
        """
        if self.df is None:
            self.load_data()
        
        unique_patients = self.df['Patient ID'].unique()
        n_patients = len(unique_patients)
        
        # First split: train+val vs test
        train_val_patients, test_patients = train_test_split(
            unique_patients, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Second split: train vs val
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=self.val_size / (1 - self.test_size),  # Adjust proportion
            random_state=self.random_state
        )
        
        self.train_patients = train_patients
        self.val_patients = val_patients
        self.test_patients = test_patients
        
        print(f"\n📊 Patient-level splits:")
        print(f"   Train: {len(train_patients):,} patients ({len(train_patients)/n_patients*100:.1f}%)")
        print(f"   Val:   {len(val_patients):,} patients ({len(val_patients)/n_patients*100:.1f}%)")
        print(f"   Test:  {len(test_patients):,} patients ({len(test_patients)/n_patients*100:.1f}%)")
        
        return train_patients, val_patients, test_patients
    
    def interpolate_patient_timeseries(self, patient_df, target_freq='1S'):
        """
        Interpolate patient vitals to uniform 1-second resolution
        Uses linear interpolation to fill gaps
        
        Args:
            patient_df: DataFrame for single patient
            target_freq: Target frequency for resampling (default: '1S' = 1 second)
        
        Returns:
            DataFrame with interpolated HR and SpO2 at uniform intervals
        """
        # Sort by timestamp
        patient_df = patient_df.sort_values('Timestamp').copy()
        patient_df.set_index('Timestamp', inplace=True)
        
        # Resample to target frequency and interpolate
        resampled = patient_df[['Heart Rate', 'Oxygen Saturation']].resample(target_freq).mean()
        interpolated = resampled.interpolate(method='linear', limit_direction='both')
        
        # Also preserve risk labels (forward fill)
        if 'Risk Category' in patient_df.columns:
            risk_resampled = patient_df['Risk Category'].resample(target_freq).ffill()
            interpolated['Risk Category'] = risk_resampled
        
        interpolated.reset_index(inplace=True)
        return interpolated
    
    def create_windows(self, timeseries_df, include_time_features=True):
        """
        Create sliding windows from interpolated time series
        
        Args:
            timeseries_df: DataFrame with columns [Timestamp, Heart Rate, Oxygen Saturation]
            include_time_features: Add sin/cos time-of-day encoding (default: True)
        
        Returns:
            windows: numpy array of shape (N, channels, window_size)
            labels: numpy array of risk labels (0=Low Risk, 1=High Risk)
            timestamps: list of timestamp arrays for each window
        """
        hr_values = timeseries_df['Heart Rate'].values
        spo2_values = timeseries_df['Oxygen Saturation'].values
        timestamps = timeseries_df['Timestamp'].values
        
        # Extract risk labels if available
        if 'Risk Category' in timeseries_df.columns:
            risk_labels = (timeseries_df['Risk Category'] == 'High Risk').astype(int).values
        else:
            risk_labels = None
        
        windows = []
        window_labels = []
        window_timestamps = []
        
        # Sliding window extraction
        for i in range(0, len(hr_values) - self.window_size + 1, self.stride):
            end_idx = i + self.window_size
            
            hr_window = hr_values[i:end_idx]
            spo2_window = spo2_values[i:end_idx]
            ts_window = timestamps[i:end_idx]
            
            # Stack as (2, window_size)
            window = np.stack([hr_window, spo2_window], axis=0)
            windows.append(window)
            window_timestamps.append(ts_window)
            
            # Use majority vote for window label
            if risk_labels is not None:
                window_label = np.median(risk_labels[i:end_idx])
                window_labels.append(int(window_label > 0.5))
            else:
                window_labels.append(0)
        
        windows = np.array(windows)  # Shape: (N, 2, window_size)
        
        # Add cyclical time features if requested
        if include_time_features and len(windows) > 0:
            # Use first timestamp of each window for encoding
            first_timestamps = [ts[0] for ts in window_timestamps]
            windows = add_cyclical_time(windows, np.array(first_timestamps))
            # Now shape: (N, 4, window_size) with HR, SpO2, sin(hour), cos(hour)
        
        return windows, np.array(window_labels), window_timestamps
    
    def get_dataset(self, split='train', include_time_features=True, normalize=True):
        """
        Get processed dataset for specified split
        
        Since each patient has only 1 snapshot, we create a synthetic time-series
        by treating records chronologically and creating windows across the dataset.
        
        Args:
            split: 'train', 'val', or 'test'
            include_time_features: Add cyclical time encoding
            normalize: Apply z-score normalization to HR and SpO2
        
        Returns:
            X: numpy array of shape (N, channels, window_size)
            y: numpy array of risk labels
            metadata: dict with patient IDs and timestamps
        """
        if self.train_patients is None:
            self.create_patient_splits()
        
        # Select patient IDs based on split
        if split == 'train':
            patient_ids = self.train_patients
        elif split == 'val':
            patient_ids = self.val_patients
        elif split == 'test':
            patient_ids = self.test_patients
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        # Filter dataframe to selected patients and sort by timestamp
        split_df = self.df[self.df['Patient ID'].isin(patient_ids)].copy()
        split_df = split_df.sort_values('Timestamp').reset_index(drop=True)
        
        print(f"\n🔧 Processing {split} split...")
        print(f"   Patients: {len(patient_ids):,}")
        print(f"   Records: {len(split_df):,}")
        
        # Since each patient has 1 snapshot, create synthetic time-series
        # by creating windows from chronologically sorted records
        hr_values = split_df['Heart Rate'].values
        spo2_values = split_df['Oxygen Saturation'].values
        timestamps = split_df['Timestamp'].values
        risk_labels = (split_df['Risk Category'] == 'High Risk').astype(int).values
        patient_ids_array = split_df['Patient ID'].values
        
        all_windows = []
        all_labels = []
        all_timestamps = []
        patient_id_map = []
        
        # Create sliding windows from the chronologically sorted data
        for i in range(0, len(hr_values) - self.window_size + 1, self.stride):
            end_idx = i + self.window_size
            
            hr_window = hr_values[i:end_idx]
            spo2_window = spo2_values[i:end_idx]
            ts_window = timestamps[i:end_idx]
            patient_window = patient_ids_array[i:end_idx]
            
            # Stack as (2, window_size)
            window = np.stack([hr_window, spo2_window], axis=0)
            all_windows.append(window)
            all_timestamps.append(ts_window)
            
            # Use majority vote for window label
            window_label = np.median(risk_labels[i:end_idx])
            all_labels.append(int(window_label > 0.5))
            
            # Track which patients are in this window (use first patient ID)
            patient_id_map.append(patient_window[0])
        
        X = np.array(all_windows)  # Shape: (N, 2, window_size)
        y = np.array(all_labels)
        
        print(f"   ✓ Generated {len(X):,} windows")
        
        # Add cyclical time features if requested
        if include_time_features and len(X) > 0:
            # Use first timestamp of each window for encoding
            first_timestamps = [ts[0] for ts in all_timestamps]
            X = add_cyclical_time(X, np.array(first_timestamps))
            print(f"   ✓ Added cyclical time features (sin/cos hour)")
        
        # Normalize HR and SpO2 channels (not time features)
        if normalize and len(X) > 0:
            # Normalize only first 2 channels (HR and SpO2)
            for ch in range(2):
                mean = X[:, ch, :].mean()
                std = X[:, ch, :].std()
                X[:, ch, :] = (X[:, ch, :] - mean) / (std + 1e-6)
            print(f"   ✓ Normalized HR and SpO2 channels")
        
        # Risk distribution
        if len(y) > 0:
            n_low = (y == 0).sum()
            n_high = (y == 1).sum()
            print(f"   ✓ Risk distribution: {n_low:,} Low Risk ({n_low/len(y)*100:.1f}%), {n_high:,} High Risk ({n_high/len(y)*100:.1f}%)")
        
        metadata = {
            'patient_ids': patient_id_map,
            'timestamps': all_timestamps,
            'split': split,
            'n_patients': len(patient_ids),
            'note': 'Windows created from chronologically sorted snapshots (not per-patient time-series)'
        }
        
        return X, y, metadata


def load_vitals_dataset(csv_path=None, window_size=256, stride=128, include_time_features=True):
    """
    Convenience function to load all splits at once
    
    Returns:
        dict with keys 'train', 'val', 'test', each containing (X, y, metadata)
    """
    loader = VitalsDataLoader(csv_path=csv_path, window_size=window_size, stride=stride)
    loader.load_data()
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        X, y, metadata = loader.get_dataset(split, include_time_features=include_time_features)
        datasets[split] = (X, y, metadata)
    
    return datasets
