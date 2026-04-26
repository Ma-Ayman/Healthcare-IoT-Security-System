import numpy as np
# =========================================================
# Time-Series Windowing Function
# =========================================================

def create_sequences(df, feature_cols, window_size=6):
    """
    Convert time-series ICU data into sliding windows for deep learning models.

    Each patient's sequential data is split into overlapping windows of fixed length.
    This allows the model to learn temporal patterns in physiological signals.
    """

    sequences = []     # Stores generated time-series windows (model inputs)
    patient_ids = []   # Stores corresponding patient IDs for each window

    # Iterate over each patient separately to preserve temporal structure
    for pid, group in df.groupby(level=0):

        # Extract only feature columns and convert to numpy array
        # Shape: (time_steps, num_features)
        values = group[feature_cols].values

        # Create sliding windows over time dimension
        # Each window contains 'window_size' consecutive time steps
        for i in range(len(values) - window_size + 1):

            # Extract a sequence of fixed length (6 timesteps)
            seq = values[i:i + window_size]

            # Append sequence to dataset
            sequences.append(seq)

            # Keep track of which patient this sequence belongs to
            patient_ids.append(pid)

    # Convert lists to numpy arrays for deep learning models
    return np.array(sequences), np.array(patient_ids)