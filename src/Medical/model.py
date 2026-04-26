from tensorflow.keras import layers, Model  # Import Keras API for building deep learning models

# =========================================================
# Build LSTM Autoencoder Model
# =========================================================

def build_lstm_autoencoder(timesteps, features):
    """
    Build an LSTM-based Autoencoder model for anomaly detection.

    Parameters:
        timesteps (int): number of time steps in each sequence
        features (int): number of features per time step

    Returns:
        model: compiled Keras model
    """
    # =========================================================
    # Input Layer
    # =========================================================
    # Define the input shape of the model:
    # (batch_size, timesteps, features)
    inputs = layers.Input(shape=(timesteps, features))
    
    # =========================================================
    # Encoder (Feature Compression)
    # =========================================================
    # First LSTM layer processes temporal patterns
    # return_sequences=True keeps sequence structure
    x = layers.LSTM(64, return_sequences=True)(inputs)
    # Second LSTM layer compresses information into latent representation
    x = layers.LSTM(32)(x)
    # At this point:
    # The model has learned a compressed representation of normal patient behavior
    
    # =========================================================
    # Repeat Vector (Bottleneck Expansion)
    # =========================================================
    # This layer repeats the compressed vector for each timestep
    # This allows the decoder to reconstruct the full sequence
    x = layers.RepeatVector(timesteps)(x)
    
    # =========================================================
    # Decoder (Reconstruction Phase)
    # =========================================================
    # First decoder LSTM expands temporal structure
    x = layers.LSTM(32, return_sequences=True)(x)
    # Second decoder LSTM restores original sequence complexity
    x = layers.LSTM(64, return_sequences=True)(x)
    
    # =========================================================
    # Output Layer
    # =========================================================
    # TimeDistributed applies Dense layer at each timestep
    # Output shape matches original feature space
    outputs = layers.TimeDistributed(
        layers.Dense(features)
    )(x)
    
    # =========================================================
    # Build Model
    # =========================================================
    # Define full autoencoder model
    model = Model(inputs, outputs)
    # Compile model using MSE loss (used for reconstruction error)
    # Lower error = normal behavior
    # Higher error = anomaly
    model.compile(optimizer='adam', loss='mse')
    
    return model