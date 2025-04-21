from model import ConvAutoEncoder, HierarchicalCAE, TemporalCAE, HTemporalCAE
import numpy as np
import keras
import os

def predict_cae(input_data, input_shape=(256, 80, 2), output_dim=64):
    """Train and predict using CAE"""
    cae = ConvAutoEncoder(
        input_shape=input_shape,
        output_dim=output_dim,
        filters=[16, 32, 64, 128],
        kernel=(3, 3),
        stride=(1, 1),
        strideundo=2,
        pool=(2, 2),
        optimizer="adamax",
        lossfn="mse"
    )

    history = cae.fit(
        input_data, 
        epochs=100,
        batch_size=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    pred = cae.ae.predict(input_data)
    mse = np.mean((input_data - pred)**2)
    print(f"\nCAE Reconstruction MSE: {mse:.6f}")
    
    return pred, mse, history

def predict_hcae(input_data, input_shape=(256, 80, 2)):
    """Train and predict using HCAE"""
    hcae = HierarchicalCAE(
        input_shape=input_shape,
        n_families=4,
        modes_per_family=72,
        filters=[16, 32, 64, 128]
    )

    histories = hcae.fit(
        input_data,
        epochs=100,
        batch_size=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    pred = hcae.predict(input_data)
    mse = np.mean((input_data - pred)**2)
    print(f"\nHCAE Reconstruction MSE: {mse:.6f}")
    
    return pred, mse, histories

def predict_pod(input_data, n_components=64):
    """Predict using POD/PCA"""
    original_shape = input_data.shape
    X = input_data[0].reshape(-1, input_data.shape[-1])
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    U, S, Vt = np.linalg.svd(X_centered.T @ X_centered, full_matrices=False)
    print(f"Top 5 singular values: {S[:5]}")
    
    U_r = U[:, :n_components]
    X_proj = X_centered @ U_r
    X_reconstructed = X_proj @ U_r.T + X_mean
    
    reconstructed = X_reconstructed.reshape(original_shape[1:])
    reconstructed = reconstructed[np.newaxis, ...]
    
    mse = np.mean((input_data - reconstructed) ** 2)
    print(f"\nPOD Reconstruction MSE ({n_components} components): {mse:.6f}")
    
    return reconstructed, mse

def predict_cae_temporal(input_data, batch_size=4, input_shape=(256, 80, 2), output_dim=64):
    """Train and predict using CAE with temporal batches"""
    # Reshape input to remove singleton dimension
    input_data = input_data.squeeze(1)  # Now shape: (n_timesteps, height, width, 2)
    
    cae = ConvAutoEncoder(
        input_shape=input_shape,
        output_dim=output_dim,
        filters=[16, 32, 64, 128],
        kernel=(3, 3),
        stride=(1, 1),
        strideundo=2,
        pool=(2, 2),
        optimizer="adamax",
        lossfn="mse"
    )

    history = cae.fit(
        input_data,
        epochs=100,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the timesteps
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Predict and calculate MSE for each timestep
    pred = cae.ae.predict(input_data, batch_size=batch_size)
    mse = np.mean((input_data - pred)**2)
    print(f"\nCAE Reconstruction MSE: {mse:.6f}")
    
    # Add singleton dimension back for consistency
    pred = pred[:, np.newaxis, ...]
    
    return pred, mse, history

def predict_convlstm(input_data, input_shape, output_dim=64, sequence_length=4, batch_size=4, return_model=False):
    """
    Train and predict using Temporal CNN Autoencoder
    
    Args:
        input_data: Shape (n_sequences, sequence_length, height, width, channels)
        input_shape: Tuple (sequence_length, height, width, channels)
        output_dim: Dimension of latent space
        sequence_length: Number of timesteps in each sequence
    """
    # Define paths for weight files
    weights_dir = './weights/temporal_cae'
    os.makedirs(weights_dir, exist_ok=True)
    
    temporal_cae = TemporalCAE(
        input_shape=input_shape,
        output_dim=output_dim,
        filters=[16, 32, 64],
        kernel=(3, 3),
        stride=(1, 1),
        pool=(2, 2),
        optimizer="adamax",
        lossfn="mse"
    )

    # Check if weights exist
    if os.path.exists(os.path.join(weights_dir, "spatial_encoder.weights.h5")):
        print("Loading existing weights...")
        temporal_cae.load_weights(weights_dir)
        
        # Predict using loaded weights
        pred = temporal_cae.predict(input_data)
        mse = np.mean((input_data - pred)**2)
        print(f"\nTemporal CAE Reconstruction MSE (loaded weights): {mse:.6f}")
        
        if return_model:
            return temporal_cae, pred, mse, None
        return pred, mse, None
    
    print("No existing weights found. Training new model...")
    print(f"Input shape: {input_data.shape}")
    
    # Train the model
    history = temporal_cae.fit(
        input_data,
        epochs=50,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # Save weights after training
    print("Saving weights...")
    temporal_cae.save_weights(weights_dir)

    # Generate predictions
    pred = temporal_cae.predict(input_data)
    mse = np.mean((input_data - pred)**2)
    print(f"\nTemporal CAE Reconstruction MSE (new training): {mse:.6f}")

    if return_model:
        return temporal_cae, pred, mse, history
    return pred, mse, history

def predict_htemporal_cae(input_data, input_shape, n_families=4, modes_per_family=64, batch_size=4, return_model=False):
    """
    Train and predict using Hierarchical Temporal CAE
    
    Args:
        input_data: Shape (n_sequences, sequence_length, height, width, channels)
        input_shape: Tuple (sequence_length, height, width, channels)
        n_families: Number of mode families
        modes_per_family: Number of modes per family
        batch_size: Batch size for training
        return_model: Whether to return the trained model
    """
    train_size = len(input_data)
    weights_dir = f'./weights/htemporal_cae_{train_size}'
    os.makedirs(weights_dir, exist_ok=True)
    
    htcae = HTemporalCAE(
        input_shape=input_shape,
        n_families=n_families,
        modes_per_family=modes_per_family,
        filters=[16, 32, 64],
        kernel=(3, 3),
        stride=(1, 1),
        pool=(2, 2)
    )

    # Check if weights exist for this training size
    if os.path.exists(os.path.join(weights_dir, "temporal_subnetwork_1_spatial_encoder.weights.h5")):
        print(f"Loading existing weights for train_size={train_size}...")
        htcae.load_weights(weights_dir)
        
        pred = htcae.predict(input_data)
        mse = np.mean((input_data - pred)**2)
        print(f"\nHTCAE Reconstruction MSE (loaded weights): {mse:.6f}")
        
        if return_model:
            return htcae, pred, mse, None
        return pred, mse, None
    
    print("No existing weights found. Training new model...")
    print(f"Input shape: {input_data.shape}")
    
    # Train the model
    histories = htcae.fit(
        input_data,
        epochs=50,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    print(f"Saving weights for train_size={train_size}...")
    htcae.save_weights(weights_dir)

    pred = htcae.predict(input_data)
    mse = np.mean((input_data - pred)**2)
    print(f"\nHTCAE Reconstruction MSE (new training): {mse:.6f}")

    if return_model:
        return htcae, pred, mse, histories
    return pred, mse, histories

def predict_future_timesteps(model, test_sequences, sequence_length=4):
    """
    Predict future timesteps using the trained model
    
    Args:
        model: Trained HTemporalCAE model
        test_sequences: Test data sequences
        sequence_length: Length of input sequences
    
    Returns:
        numpy.ndarray: Predicted sequences
    """
    print(f"Predicting future timesteps for {len(test_sequences)} sequences...")
    predictions = model.predict(test_sequences)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Test MSE: {np.mean((test_sequences - predictions) ** 2):.6f}")
    
    return predictions