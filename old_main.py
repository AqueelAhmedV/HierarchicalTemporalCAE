from predict import predict_cae, predict_hcae, predict_pod, predict_cae_temporal, predict_htemporal_cae, predict_future_timesteps, predict_convlstm
from preprocess import read_csv_to_dataset, prepare_for_autoencoder, \
prepare_timestep_data, timestep_data_generator

from plotting import plot_velocity_field, plot_training_history, plot_hcae_history, compare_predictions, plot_temporal_reconstructions, plot_temporal_mse, plot_temporal_comparison, plot_temporal_error
import numpy as np
import os

# Configuration parameters
input_shape = (256, 80, 2)
output_dim = 64
filters = [16, 32, 64, 128]
kernel = (3, 3)
stride = (1, 1)
strideundo = 2
pool = (2, 2)

def main():
    print("here")
    # Load and prepare data
    dataset = read_csv_to_dataset('data/csv/U_data.csv')
    input_data = prepare_for_autoencoder(dataset)

    # Check if cached data exists
    cache_file = './data/processed/temporal_data_new.npz'
    
    if os.path.exists(cache_file):
        print("Loading cached data...")
        cached_data = np.load(cache_file)
        data_array = cached_data['data']
        timesteps = cached_data['timesteps']
    else:
        all_data = []
        all_timesteps = []
        
        # Use the generator to process data in chunks
        for timestep, data in timestep_data_generator('./data/csv/timestamp_data_full.csv'):
            processed_data = prepare_for_autoencoder(data)
            all_data.append(processed_data)
            all_timesteps.append(timestep)
            print(f"Processed timestep {timestep}")
        
        # Convert to numpy arrays
        data_array = np.concatenate(all_data, axis=0)
        timesteps = np.array(all_timesteps)
        
        # Sort by timestep
        sort_idx = np.argsort(timesteps)
        data_array = data_array[sort_idx]
        timesteps = timesteps[sort_idx]
        
        # Save processed data
        print("Saving processed data to cache...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, data=data_array, timesteps=timesteps)
    
    # Print shape for debugging
    print(f"Data array shape: {data_array.shape}")
    
    # Split data into training and test sets
    train_size = 80
    train_data = data_array[:train_size]
    test_data = data_array[train_size:]
    train_timesteps = timesteps[:train_size]
    test_timesteps = timesteps[train_size:]

    # Create sequences for training
    sequence_length = 4
    train_sequences = []
    for i in range(len(train_data) - sequence_length + 1):
        sequence = train_data[i:i+sequence_length]
        train_sequences.append(sequence)
    train_sequences = np.array(train_sequences)

    # Create sequences for testing
    test_sequences = []
    for i in range(len(test_data) - sequence_length + 1):
        sequence = test_data[i:i+sequence_length]
        test_sequences.append(sequence)
    test_sequences = np.array(test_sequences)

    print(f"\nTrain sequences shape: {train_sequences.shape}")
    print(f"Test sequences shape: {test_sequences.shape}")

    # Define temporal shape for HTemporalCAE
    temporal_shape = (sequence_length, *input_shape)

    # Train both models and get predictions
    htcae_model, htcae_train_pred, htcae_train_mse, htcae_histories = predict_htemporal_cae(
        input_data=train_sequences,
        input_shape=temporal_shape,
        n_families=4,
        modes_per_family=64,
        batch_size=12,
        return_model=True
    )

    convlstm_model, convlstm_train_pred, convlstm_train_mse, convlstm_histories = predict_convlstm(
        input_data=train_sequences,
        input_shape=temporal_shape,
        output_dim=64,
        sequence_length=sequence_length,
        batch_size=12,
        return_model=True
    )

    # Make predictions on test data
    htcae_test_pred = htcae_model.predict(test_sequences)
    convlstm_test_pred = convlstm_model.predict(test_sequences)

    # Compare reconstructions and errors for a single timestep
    compare_predictions(
        original=train_sequences[0, 0],
        pred1=htcae_train_pred[0, 0],
        pred2=convlstm_train_pred[0, 0],
        title1="HTCAE Reconstruction",
        title2="ConvLSTM Reconstruction",
        plot_type="field"
    )

    # Compare predictions and errors for a single timestep
    compare_predictions(
        original=test_sequences[0, 0],
        pred1=htcae_test_pred[0, 0],
        pred2=convlstm_test_pred[0, 0],
        title1="HTCAE Prediction",
        title2="ConvLSTM Prediction",
        plot_type="field"
    )

    compare_predictions(
        original=train_sequences[0, 0],
        pred1=htcae_train_pred[0, 0],
        pred2=convlstm_train_pred[0, 0],
        title1="HTCAE Reconstruction",
        title2="ConvLSTM Reconstruction",
        plot_type="error"
    )

    # Compare predictions and errors for a single timestep
    compare_predictions(
        original=test_sequences[0, 0],
        pred1=htcae_test_pred[0, 0],
        pred2=convlstm_test_pred[0, 0],
        title1="HTCAE Prediction",
        title2="ConvLSTM Prediction",
        plot_type="error"
    )

    # Print MSE for both models
    print(f"\nReconstruction Results:")
    print(f"HTCAE Training MSE: {htcae_train_mse:.6f}")
    print(f"ConvLSTM Training MSE: {convlstm_train_mse:.6f}")
    
    htcae_test_mse = np.mean((test_sequences - htcae_test_pred) ** 2)
    convlstm_test_mse = np.mean((test_sequences - convlstm_test_pred) ** 2)
    print(f"\nPrediction Results:")
    print(f"HTCAE Test MSE: {htcae_test_mse:.6f}")
    print(f"ConvLSTM Test MSE: {convlstm_test_mse:.6f}")

if __name__ == "__main__":
    main()