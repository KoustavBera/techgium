"""
Cyclical time encoding for temporal features
Captures circadian rhythm patterns using sine/cosine transformations
"""

import numpy as np
import pandas as pd


def add_cyclical_time(data, timestamps):
    """
    Add cyclical time-of-day features to time series data
    
    Args:
        data: numpy array of shape (N, channels, timesteps) or (N, timesteps, channels)
            e.g., (100, 2, 256) for 100 samples with HR+SpO2 over 256 timesteps
        timestamps: pandas Series or numpy array of datetime objects
            Shape: (N,) for per-sample timestamp, or (N, timesteps) for per-timestep
    
    Returns:
        numpy array with time features appended: (N, channels+2, timesteps)
        New channels are sin(hour) and cos(hour) for circadian encoding
    
    Example:
        >>> hr_spo2 = np.random.randn(100, 2, 256)  # 100 samples, 2 channels, 256 timesteps
        >>> timestamps = pd.date_range('2024-01-01', periods=100, freq='1min')
        >>> enhanced = add_cyclical_time(hr_spo2, timestamps)
        >>> enhanced.shape
        (100, 4, 256)  # Now includes sin(hour) and cos(hour)
    """
    # Handle different timestamp formats
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.values
    
    # Convert to pandas datetime if needed
    if not isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
        timestamps = pd.to_datetime(timestamps)
    
    # Determine data shape and format
    if data.ndim == 3:
        n_samples, dim1, dim2 = data.shape
        # Assume format is (N, channels, timesteps) if dim1 < dim2, else (N, timesteps, channels)
        if dim1 < dim2:
            channels_first = True
            n_channels = dim1
            n_timesteps = dim2
        else:
            channels_first = False
            n_channels = dim2
            n_timesteps = dim1
    else:
        raise ValueError(f"Expected 3D data array, got shape {data.shape}")
    
    # Extract hour information
    if timestamps.ndim == 1 and len(timestamps) == n_samples:
        # Single timestamp per sample - broadcast to all timesteps
        hours = pd.Series(timestamps).dt.hour + pd.Series(timestamps).dt.minute / 60.0
        hours = hours.values
        
        # Create time features for each sample
        sin_time = np.sin(2 * np.pi * hours / 24)
        cos_time = np.cos(2 * np.pi * hours / 24)
        
        # Broadcast to all timesteps
        if channels_first:
            sin_time_expanded = np.tile(sin_time[:, np.newaxis, np.newaxis], (1, 1, n_timesteps))
            cos_time_expanded = np.tile(cos_time[:, np.newaxis, np.newaxis], (1, 1, n_timesteps))
            time_features = np.concatenate([sin_time_expanded, cos_time_expanded], axis=1)
            result = np.concatenate([data, time_features], axis=1)
        else:
            sin_time_expanded = np.tile(sin_time[:, np.newaxis, np.newaxis], (1, n_timesteps, 1))
            cos_time_expanded = np.tile(cos_time[:, np.newaxis, np.newaxis], (1, n_timesteps, 1))
            time_features = np.concatenate([sin_time_expanded, cos_time_expanded], axis=2)
            result = np.concatenate([data, time_features], axis=2)
    
    elif timestamps.ndim == 2 and timestamps.shape == (n_samples, n_timesteps):
        # Per-timestep timestamps
        hours = np.zeros((n_samples, n_timesteps))
        for i in range(n_samples):
            ts_series = pd.Series(timestamps[i])
            hours[i] = ts_series.dt.hour + ts_series.dt.minute / 60.0
        
        sin_time = np.sin(2 * np.pi * hours / 24)
        cos_time = np.cos(2 * np.pi * hours / 24)
        
        if channels_first:
            # Reshape to (N, 1, timesteps) and concatenate
            time_features = np.stack([sin_time, cos_time], axis=1)
            result = np.concatenate([data, time_features], axis=1)
        else:
            # Reshape to (N, timesteps, 1) and concatenate
            time_features = np.stack([sin_time, cos_time], axis=2)
            result = np.concatenate([data, time_features], axis=2)
    else:
        raise ValueError(f"Timestamp shape {timestamps.shape} incompatible with data shape {data.shape}")
    
    return result


def get_cyclical_features(timestamp):
    """
    Extract cyclical time features from a single timestamp
    
    Args:
        timestamp: datetime-like object
    
    Returns:
        tuple: (sin_hour, cos_hour)
    
    Example:
        >>> import pandas as pd
        >>> ts = pd.Timestamp('2024-01-01 14:30:00')  # 2:30 PM
        >>> sin_h, cos_h = get_cyclical_features(ts)
        >>> print(f"Hour encoding: sin={sin_h:.3f}, cos={cos_h:.3f}")
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    
    hour = timestamp.hour + timestamp.minute / 60.0
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    
    return sin_hour, cos_hour


def add_cyclical_day_of_week(data, timestamps):
    """
    Add day-of-week cyclical encoding (optional - for weekly patterns)
    
    Args:
        data: numpy array of shape (N, channels, timesteps)
        timestamps: datetime array
    
    Returns:
        numpy array with 2 additional channels: sin(day_of_week), cos(day_of_week)
    """
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.values
    
    if not isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
        timestamps = pd.to_datetime(timestamps)
    
    # Get day of week (0=Monday, 6=Sunday)
    day_of_week = pd.Series(timestamps).dt.dayofweek.values
    
    sin_day = np.sin(2 * np.pi * day_of_week / 7)
    cos_day = np.cos(2 * np.pi * day_of_week / 7)
    
    # Broadcast to shape (N, 2, timesteps)
    n_samples = data.shape[0]
    n_timesteps = data.shape[2] if data.shape[1] < data.shape[2] else data.shape[1]
    
    sin_day_expanded = np.tile(sin_day[:, np.newaxis, np.newaxis], (1, 1, n_timesteps))
    cos_day_expanded = np.tile(cos_day[:, np.newaxis, np.newaxis], (1, 1, n_timesteps))
    
    time_features = np.concatenate([sin_day_expanded, cos_day_expanded], axis=1)
    result = np.concatenate([data, time_features], axis=1)
    
    return result
