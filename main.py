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

    # Create plots for different training sizes
    training_sizes = [50, 80, 120]
    
    for train_size in training_sizes:
        # Split data into training and test sets for this size
        train_data = data_array[:train_size]
        test_data = data_array[train_size:]
        train_timesteps = timesteps[:train_size]
        test_timesteps = timesteps[train_size:]

        # Create sequences for this training size
        train_sequences = []
        for i in range(len(train_data) - sequence_length + 1):
            sequence = train_data[i:i+sequence_length]
            train_sequences.append(sequence)
        train_sequences = np.array(train_sequences)

        # Train model with this training size
        htcae_model, htcae_train_pred, htcae_train_mse, htcae_histories = predict_htemporal_cae(
            input_data=train_sequences,
            input_shape=temporal_shape,
            n_families=4,
            modes_per_family=64,
            batch_size=12,
            return_model=True
        )
        
        # Plot training results
        plot_velocity_field(
            original=train_sequences[-1][-1],
            reconstructed=htcae_train_pred[-1][-1],
            title=f"Training Size {train_size} - Training",
            save_path=f"./figures/size_{train_size}_train_comparison.png"
        )
        
        # Get and plot test predictions
        test_pred = htcae_model.predict(test_sequences[:100])
        plot_velocity_field(
            original=test_sequences[0][-1],  # First test sequence, last timestep
            reconstructed=test_pred[0][-1],
            title=f"Training Size {train_size} - Test",
            save_path=f"./figures/size_{train_size}_test_comparison.png"
        )
        
        # Print metrics
        print(f"\nResults for training size {train_size}:")
        print(f"Training MSE: {htcae_train_mse:.6f}")
        test_mse = np.mean((test_sequences[:100] - test_pred) ** 2)
        print(f"Test MSE: {test_mse:.6f}")

if __name__ == "__main__":
    main()