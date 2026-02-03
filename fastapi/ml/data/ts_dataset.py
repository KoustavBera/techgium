import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, window_size=256, normalize=True):
        """
        Dataset for time series sequences with optional normalization
        
        Args:
            sequences: List of numpy arrays or single array of shape (N, channels, timesteps)
            window_size: Expected window size for validation (default: 256)
            normalize: Whether to apply per-feature normalization (default: True)
        """
        self.window_size = window_size
        self.data = []

        # Handle different input formats
        if isinstance(sequences, np.ndarray):
            # Already in array format
            if sequences.ndim == 3:
                # Shape: (N, channels, timesteps)
                for i in range(len(sequences)):
                    seq = sequences[i]
                    if seq.shape[-1] >= window_size:
                        seq = self._normalize(seq) if normalize else seq
                        self.data.append(seq)
            elif sequences.ndim == 2:
                # Shape: (timesteps, channels) - single sequence
                seq = sequences
                if len(seq) >= window_size:
                    seq = self._normalize(seq) if normalize else seq
                    self.data.append(seq)
        elif isinstance(sequences, list):
            # List of sequences
            for seq in sequences:
                if isinstance(seq, np.ndarray):
                    seq = self._normalize(seq) if normalize else seq
                    if seq.shape[-1] >= window_size:
                        self.data.append(seq)
                else:
                    raise ValueError(f"Expected numpy array, got {type(seq)}")
        else:
            raise ValueError(f"Expected list or numpy array, got {type(sequences)}")

    def _normalize(self, x):
        """
        Apply per-feature z-score normalization
        
        Args:
            x: Array of shape (channels, timesteps) or (timesteps, channels)
        
        Returns:
            Normalized array
        """
        # Determine channel dimension
        if x.ndim == 2:
            # Assume (channels, timesteps) if first dim is smaller
            if x.shape[0] < x.shape[1]:
                channels_first = True
                n_channels = x.shape[0]
            else:
                channels_first = False
                n_channels = x.shape[1]
            
            # Normalize each channel independently
            x_norm = x.copy()
            for ch in range(n_channels):
                if channels_first:
                    channel_data = x[ch, :]
                    mean = channel_data.mean()
                    std = channel_data.std()
                    x_norm[ch, :] = (channel_data - mean) / (std + 1e-6)
                else:
                    channel_data = x[:, ch]
                    mean = channel_data.mean()
                    std = channel_data.std()
                    x_norm[:, ch] = (channel_data - mean) / (std + 1e-6)
            
            return x_norm
        else:
            return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        
        # Ensure correct shape: (channels, timesteps)
        if x.ndim == 2:
            # Check if we need to transpose
            if x.shape[0] > x.shape[1]:
                # Likely (timesteps, channels) - transpose
                x = x.T
        
        # Convert to tensor and add batch dimension for compatibility
        return torch.tensor(x, dtype=torch.float32)
        