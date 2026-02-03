import numpy as np

def generate_hr_spo2_sequence(length=1024, anomaly=False):
    t = np.linspace(0, 10, length)

    # Normal baseline signals
    hr = 70 + 5 * np.sin(2 * np.pi * 0.2 * t)
    spo2 = 97 + 0.5 * np.sin(2 * np.pi * 0.1 * t)

    if anomaly:
        # Adjust anomaly window based on sequence length
        if length < 400:
            # For shorter sequences (like 256), use proportional positioning
            start = np.random.randint(int(length * 0.25), int(length * 0.5))
            duration = np.random.randint(int(length * 0.15), int(length * 0.3))
        else:
            # For longer sequences, use original logic
            start = np.random.randint(300, 500)
            duration = np.random.randint(80, 150)
        
        end = min(start + duration, length)  # Ensure we don't exceed bounds
        
        # 1️⃣ Sudden heart-rate spike (event)
        hr[start:end] += np.random.uniform(15, 25)

        # 2️⃣ Temporary oxygen drop (desaturation)
        spo2[start:end] -= np.random.uniform(4, 7)

        # 3️⃣ Extra rhythm instability
        hr += 3 * np.sin(2 * np.pi * 1.5 * t)

    # Measurement noise (always present)
    hr += np.random.normal(0, 1, length)
    spo2 += np.random.normal(0, 0.2, length)

    return np.stack([hr, spo2], axis=1)
    