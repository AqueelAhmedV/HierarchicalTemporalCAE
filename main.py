# --- main.py (Corrected Optimizer Instantiation) ---

# Keep original imports for data processing and plotting
from plotting import (
    plot_velocity_field,
    # plot_training_history,
    # plot_hcae_history,
    # compare_predictions,
    # plot_temporal_reconstructions,
    # plot_temporal_mse,
    # plot_temporal_comparison,
    # plot_temporal_error,
)
from preprocess import (
    # read_csv_to_dataset,
    prepare_for_autoencoder,
    # prepare_timestep_data,
    timestep_data_generator,
)

# Import the new model builder
from models import build_htcae_subnetwork  # Assuming models.py exists

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import math  # Keep math import

# --- Configuration parameters ---
INPUT_SHAPE_SPATIAL = (256, 80, 2)
LATENT_DIM = 64
N_FAMILIES = 4  # Number of hierarchical levels
MODES_PER_FAMILY = 64  # Latent dim per subnetwork
BATCH_SIZE = 12
EPOCHS = 100  # Adjust as needed
PATIENCE_EARLY_STOPPING = 10
PATIENCE_LR_REDUCTION = 5
# Removed global OPTIMIZER definition
LOSS = "mse"
SEQUENCE_LENGTH = 4  # Defined here for clarity, used multiple places


# --- Helper Function ---
def predict_with_hierarchy(models_list, initial_input_sequence):
    """
    Performs prediction using the trained hierarchical model list.

    Args:
        models_list: A list of trained Keras subnetwork models.
        initial_input_sequence: The initial input sequence(s) for the first model.

    Returns:
        The final reconstructed sequence(s).
    """
    cumulative_prediction = np.zeros_like(initial_input_sequence)
    current_input = initial_input_sequence

    for i, model in enumerate(models_list):
        print(f"Predicting with subnetwork {i + 1}/{len(models_list)}...")
        # Input for this level is the original minus previous reconstructions
        current_input_for_model = initial_input_sequence - cumulative_prediction
        sub_prediction = model.predict(current_input_for_model, batch_size=BATCH_SIZE)
        cumulative_prediction += sub_prediction

    return cumulative_prediction


# --- Main Function ---
def main():
    cache_file = "./data/processed/temporal_data_new.npz"
    if os.path.exists(cache_file):
        print("Loading cached data...")
        cached_data = np.load(cache_file)
        data_array = cached_data["data"]
        timesteps = cached_data["timesteps"]
        print(f"Loaded data shape: {data_array.shape}")
    else:
        print("Processing data from CSV...")
        all_data, all_timesteps = [], []
        generator = timestep_data_generator("./data/csv/timestamp_data_full.csv")
        count = 0
        for timestep, data in generator:
            processed_data = prepare_for_autoencoder(data)
            all_data.append(processed_data[np.newaxis, ...])
            all_timesteps.append(timestep)
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} timesteps...")
        if not all_data:
            print("Error: No data loaded from generator.")
            return

        data_array = np.concatenate(all_data, axis=0)
        timesteps = np.array(all_timesteps)
        print("Sorting data...")
        sort_idx = np.argsort(timesteps)
        data_array = data_array[sort_idx]
        timesteps = timesteps[sort_idx]
        print(f"Final processed data shape: {data_array.shape}")
        print("Caching data...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, data=data_array, timesteps=timesteps)
        print(f"Data saved to {cache_file}")

    print(f"Full data array shape: {data_array.shape}")
    print(f"Number of timesteps available: {len(timesteps)}\n")

    sequence_length = SEQUENCE_LENGTH
    temporal_shape_build = (sequence_length, *INPUT_SHAPE_SPATIAL)

    training_sizes = [50, 80, 120]  # Use one size for faster testing

    for train_size in training_sizes:
        print(f"\n--- Processing for Training Size: {train_size} ---")
        if train_size >= len(data_array) - sequence_length:
            print(
                f"Warning: train_size {train_size} too large for available data ({len(data_array)}). Skipping."
            )
            continue

        train_data = data_array[:train_size]
        val_split_index = max(sequence_length, int(train_size * 0.8))
        train_data_actual = data_array[:val_split_index]
        val_data = data_array[val_split_index:train_size]
        test_data = data_array[train_size:]

        print(f"Train data shape: {train_data_actual.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        print("Preparing sequences...")
        original_train_sequences = np.array(
            [
                train_data_actual[i : i + sequence_length]
                for i in range(len(train_data_actual) - sequence_length + 1)
            ]
        )
        if len(val_data) >= sequence_length:
            validation_sequences = np.array(
                [
                    val_data[i : i + sequence_length]
                    for i in range(len(val_data) - sequence_length + 1)
                ]
            )
            val_data_for_fit = (validation_sequences, validation_sequences)
        else:
            print(
                "Warning: Not enough validation data to create sequences, using validation_split instead."
            )
            validation_sequences = None
            val_data_for_fit = 0.2

        test_sequences = np.array(
            [
                test_data[i : i + sequence_length]
                for i in range(len(test_data) - sequence_length + 1)
            ]
        )
        test_sequences_eval = test_sequences[:100]

        print(f"Training sequences shape: {original_train_sequences.shape}")
        if validation_sequences is not None:
            print(f"Validation sequences shape: {validation_sequences.shape}")
        print(f"Test sequences shape (evaluation subset): {test_sequences_eval.shape}")

        # --- Hierarchical Training Loop ---
        cumulative_train_prediction = np.zeros_like(original_train_sequences)
        trained_models = []
        all_histories = {}

        for i in range(N_FAMILIES):
            model_name = f"htcae_subnetwork_{i + 1}_size_{train_size}"
            checkpoint_path = f"./models/{model_name}.keras"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            if os.path.exists(checkpoint_path):
                print(f"→ Found existing model for {model_name}, loading from disk.")
                best_model = keras.models.load_model(checkpoint_path)
            else:
                print(f"→ No saved model for {model_name}, training from scratch.")

                # Build & compile
                htcae_subnetwork = build_htcae_subnetwork(
                    input_shape_spatial=INPUT_SHAPE_SPATIAL,
                    sequence_length=SEQUENCE_LENGTH,
                    latent_dim=MODES_PER_FAMILY,
                    name=model_name,
                )
                optimizer_instance = keras.optimizers.Adamax()
                htcae_subnetwork.compile(optimizer=optimizer_instance, loss=LOSS)

                # EarlyStopping & LR reduction only
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=PATIENCE_EARLY_STOPPING,
                        restore_best_weights=True,
                        verbose=1,
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=PATIENCE_LR_REDUCTION,
                        verbose=1,
                    ),
                ]

                # Prepare input (residual)
                input_for_current_fit = (
                    original_train_sequences - cumulative_train_prediction
                )

                # Train
                history = htcae_subnetwork.fit(
                    input_for_current_fit,
                    input_for_current_fit,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=val_data_for_fit
                    if isinstance(val_data_for_fit, tuple)
                    else None,
                    validation_split=val_data_for_fit
                    if isinstance(val_data_for_fit, float)
                    else None,
                    callbacks=callbacks,
                    verbose=1,
                )
                all_histories[model_name] = history.history

                # Save the fully trained model
                print(f"→ Saving trained model to {checkpoint_path}")
                htcae_subnetwork.save(checkpoint_path)
                best_model = htcae_subnetwork

            # Predict the sub‐network's contribution
            print(f"Predicting contribution of {model_name} …")
            input_for_prediction = (
                original_train_sequences - cumulative_train_prediction
            )
            sub_pred = best_model.predict(input_for_prediction, batch_size=BATCH_SIZE)

            # Accumulate residual
            cumulative_train_prediction += sub_pred
            trained_models.append(best_model)

        # --- End Hierarchical Training Loop ---

        htcae_final_train_pred = cumulative_train_prediction
        htcae_train_mse = np.mean(
            (original_train_sequences - htcae_final_train_pred) ** 2
        )

        print("\nPredicting on Test Set...")
        test_pred = predict_with_hierarchy(trained_models, test_sequences_eval)

        print("Plotting results...")
        plot_velocity_field(
            original=original_train_sequences[-1][-1],
            reconstructed=htcae_final_train_pred[-1][-1],
            title=f"HTCAE Train Size {train_size} - Final Train Reconstruction",
            save_path=f"./figures/htcae_size_{train_size}_train_comparison.png",
        )
        plot_velocity_field(
            original=test_sequences_eval[0][-1],
            reconstructed=test_pred[0][-1],
            title=f"HTCAE Train Size {train_size} - Test Reconstruction (Seq 0)",
            save_path=f"./figures/htcae_size_{train_size}_test_comparison.png",
        )

        test_mse = np.mean((test_sequences_eval - test_pred) ** 2)

        print(f"\n--- Results for Training Size {train_size} ---")
        print(f"Final Training MSE: {htcae_train_mse:.6f}")
        print(
            f"Final Test MSE (on {len(test_sequences_eval)} sequences): {test_mse:.6f}"
        )
        print("-" * 40)


if __name__ == "__main__":
    main()
