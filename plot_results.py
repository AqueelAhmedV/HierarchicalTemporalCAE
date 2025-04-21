import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import math

# Keep original imports for data processing if needed elsewhere,
# but plotting functions are redefined or adapted below.
# from plotting import (
#     plot_velocity_field,
# )
from preprocess import (
    prepare_for_autoencoder,
    timestep_data_generator,
)

# Import the new model builder (needed for loading custom layers if any, though load_model usually handles it)
# from models import build_htcae_subnetwork # Usually not strictly needed if models were saved correctly

# --- Configuration parameters (MUST MATCH TRAINING) ---
INPUT_SHAPE_SPATIAL = (256, 80, 2)
# LATENT_DIM = 64 # Not directly needed for loading/prediction
N_FAMILIES = 4  # Number of hierarchical levels
# MODES_PER_FAMILY = 64 # Not directly needed
BATCH_SIZE = 12  # Used for prediction efficiency
# EPOCHS = 100 # Not needed
# PATIENCE_EARLY_STOPPING = 10 # Not needed
# PATIENCE_LR_REDUCTION = 5 # Not needed
# LOSS = "mse" # Not needed for prediction, but useful context
SEQUENCE_LENGTH = 4
MODEL_DIR = "./models"  # Directory where models are saved
FIGURE_DIR = "./figures_evaluation"  # Directory to save new plots
CACHE_FILE = "./data/processed/temporal_data_new.npz"
CSV_FILE = "./data/csv/timestamp_data_full.csv"

# Evaluation parameters
NUM_TEST_SEQUENCES_EVAL = 100  # Number of test sequences to evaluate on


# --- Helper Function ---
def predict_with_hierarchy(models_list, initial_input_sequence, batch_size):
    """
    Performs prediction using the trained hierarchical model list.

    Args:
        models_list: A list of loaded Keras subnetwork models.
        initial_input_sequence: The initial input sequence(s) for the first model.
        batch_size: Prediction batch size.

    Returns:
        The final reconstructed sequence(s).
    """
    if not models_list:
        print("Warning: No models provided for prediction.")
        return np.zeros_like(initial_input_sequence)

    cumulative_prediction = np.zeros_like(initial_input_sequence)
    current_input = initial_input_sequence  # Keep track for logging if needed

    for i, model in enumerate(models_list):
        print(f"Predicting with subnetwork {i + 1}/{len(models_list)}...")
        # Input for this level is the original minus previous reconstructions
        current_input_for_model = initial_input_sequence - cumulative_prediction
        # Use predict method for inference
        sub_prediction = model.predict(
            current_input_for_model, batch_size=batch_size, verbose=0
        )
        cumulative_prediction += sub_prediction

    return cumulative_prediction


# --- Plotting Functions (Adapted from example) ---
def plot_training_size_vs_overall_mse(training_sizes, overall_mse_list, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, overall_mse_list, marker="o")
    plt.title("Overall Test MSE vs. Training Size")
    plt.xlabel("Training Size (# Timesteps)")
    plt.ylabel("Mean Squared Error (MSE) on Test Set")
    plt.grid(True)
    plt.xticks(training_sizes)  # Ensure ticks are exactly at the specified sizes
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_prediction_mse_for_specific_timesteps(
    sequence_indices, mse_per_sequence, train_size, save_path
):
    plt.figure(figsize=(10, 6))
    plt.plot(sequence_indices, mse_per_sequence, marker="o")
    plt.title(f"Prediction MSE per Sequence (Test Set) - Training Size {train_size}")
    plt.xlabel("Test Sequence Index (Proxy for Timestep Further from Training)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_varying_training_size_with_mse_per_timestep(
    sequence_indices, mse_dict, save_path
):
    plt.figure(figsize=(10, 6))
    sorted_train_sizes = sorted(mse_dict.keys())
    for size in sorted_train_sizes:
        if size in mse_dict:
            plt.plot(
                sequence_indices,
                mse_dict[size],
                marker="o",
                label=f"Training Size {size}",
            )
    plt.title("Prediction MSE per Sequence for Varying Training Sizes")
    plt.xlabel("Test Sequence Index (Proxy for Timestep Further from Training)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    plt.close()


# --- Main Function ---
def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)  # Ensure figure directory exists

    # --- Data Loading and Preparation ---
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        cached_data = np.load(CACHE_FILE)
        data_array = cached_data["data"]
        timesteps = cached_data["timesteps"]
        print(f"Loaded data shape: {data_array.shape}")
    else:
        print(f"Processing data from CSV: {CSV_FILE}...")
        all_data, all_timesteps = [], []
        generator = timestep_data_generator(CSV_FILE)
        count = 0
        for timestep, data in generator:
            processed_data = prepare_for_autoencoder(data)
            all_data.append(processed_data[np.newaxis, ...])  # Add time dimension
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
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        np.savez(CACHE_FILE, data=data_array, timesteps=timesteps)
        print(f"Data saved to {CACHE_FILE}")

    print(f"Full data array shape: {data_array.shape}")
    print(f"Number of timesteps available: {len(timesteps)}\n")

    # Define training sizes to evaluate
    training_sizes = [50, 80, 120]  # Example sizes, adjust as needed

    # Store results
    results_overall_mse = {}  # Key: train_size, Value: overall test MSE
    results_per_sequence_mse = {}  # Key: train_size, Value: list of MSEs for each test sequence

    # --- Evaluation Loop ---
    for train_size in training_sizes:
        print(f"\n--- Evaluating for Training Size: {train_size} ---")

        # --- Data Splitting ---
        if train_size >= len(data_array) - SEQUENCE_LENGTH:
            print(
                f"Warning: train_size {train_size} too large for available data ({len(data_array)}). Skipping."
            )
            continue

        # Define test data based on train_size
        # Test data starts immediately after the training data ends
        test_data_start_index = train_size
        test_data = data_array[test_data_start_index:]

        if len(test_data) < SEQUENCE_LENGTH:
            print(
                f"Warning: Not enough data for test sequences (need {SEQUENCE_LENGTH}, have {len(test_data)}). Skipping train_size {train_size}."
            )
            continue

        print(f"Test data shape: {test_data.shape}")

        # --- Prepare Test Sequences ---
        print("Preparing test sequences...")
        test_sequences = np.array(
            [
                test_data[i : i + SEQUENCE_LENGTH]
                for i in range(len(test_data) - SEQUENCE_LENGTH + 1)
            ]
        )

        # Limit the number of test sequences for evaluation if needed
        num_available_test_seq = len(test_sequences)
        eval_count = min(NUM_TEST_SEQUENCES_EVAL, num_available_test_seq)
        if eval_count == 0:
            print(
                f"Warning: No test sequences could be generated for train_size {train_size}. Skipping."
            )
            continue

        test_sequences_eval = test_sequences[:eval_count]
        print(f"Test sequences shape (evaluation subset): {test_sequences_eval.shape}")

        # --- Load Trained Models ---
        trained_models = []
        models_loaded_successfully = True
        for i in range(N_FAMILIES):
            model_name = f"htcae_subnetwork_{i + 1}_size_{train_size}"
            checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}.keras")

            if os.path.exists(checkpoint_path):
                print(f"→ Loading model: {checkpoint_path}")
                try:
                    # Load model - might need custom_objects if you have custom layers/losses not standard in TF/Keras
                    model = keras.models.load_model(
                        checkpoint_path
                    )  # , custom_objects={...} )
                    trained_models.append(model)
                except Exception as e:
                    print(f"Error loading model {checkpoint_path}: {e}")
                    models_loaded_successfully = False
                    break
            else:
                print(
                    f"Error: Model file not found: {checkpoint_path}. Cannot evaluate this training size."
                )
                models_loaded_successfully = False
                break

        if not models_loaded_successfully or len(trained_models) != N_FAMILIES:
            print(
                f"Skipping evaluation for train_size {train_size} due to missing or incomplete models."
            )
            continue

        # --- Prediction on Test Set ---
        print(f"\nPredicting on {eval_count} test sequences...")
        test_pred = predict_with_hierarchy(
            trained_models, test_sequences_eval, BATCH_SIZE
        )

        # --- Calculate MSE ---
        # Overall MSE across all evaluated test sequences
        test_mse_overall = np.mean((test_sequences_eval - test_pred) ** 2)

        # MSE per sequence (average over sequence length, spatial dimensions, channels)
        # Shape: (num_sequences, seq_len, height, width, channels)
        # Axis (1, 2, 3, 4) corresponds to seq_len, height, width, channels
        test_mse_per_sequence = np.mean(
            (test_sequences_eval - test_pred) ** 2, axis=(1, 2, 3, 4)
        )

        results_overall_mse[train_size] = test_mse_overall
        results_per_sequence_mse[train_size] = (
            test_mse_per_sequence.tolist()
        )  # Store as list

        print(f"\n--- Results for Training Size {train_size} ---")
        print(f"Overall Test MSE (on {eval_count} sequences): {test_mse_overall:.6f}")
        # print(f"Test MSE per sequence (first 5): {[f'{mse:.6f}' for mse in test_mse_per_sequence[:5]]}")
        print("-" * 50)

    # --- End Evaluation Loop ---

    # --- Generate Plots ---
    print("\n--- Generating Final Plots ---")

    if not results_overall_mse:
        print("No evaluation results were generated. Cannot create plots.")
        return

    # Ensure consistent sequence indices for plotting across different training sizes
    # Use the minimum number of evaluated sequences across all sizes if they differ,
    # or simply use the indices based on NUM_TEST_SEQUENCES_EVAL (assuming it was achievable for all)
    # Here, let's assume we want to plot up to NUM_TEST_SEQUENCES_EVAL indices
    min_seq_len = NUM_TEST_SEQUENCES_EVAL  # Default assumption
    if results_per_sequence_mse:
        min_seq_len = min(len(v) for v in results_per_sequence_mse.values())
        if min_seq_len < NUM_TEST_SEQUENCES_EVAL:
            print(
                f"Warning: Plotting MSE per sequence only up to index {min_seq_len - 1} due to limited test data for some training sizes."
            )

    plot_sequence_indices = np.arange(min_seq_len)

    # Filter results to match the plotted sequence length
    filtered_mse_per_sequence = {
        size: mse_list[:min_seq_len]
        for size, mse_list in results_per_sequence_mse.items()
        if len(mse_list) >= min_seq_len  # Ensure the size had enough sequences
    }
    filtered_training_sizes_overall = sorted(
        [
            size
            for size in results_overall_mse.keys()
            if size in filtered_mse_per_sequence
        ]
    )  # Sizes that have per-sequence data
    filtered_overall_mse_list = [
        results_overall_mse[size] for size in filtered_training_sizes_overall
    ]

    if not filtered_training_sizes_overall:
        print(
            "Could not generate plots as no training size had sufficient test sequences."
        )
        return

    # Plot 1: Overall Test MSE vs Training Size
    plot_training_size_vs_overall_mse(
        filtered_training_sizes_overall,
        filtered_overall_mse_list,
        save_path=os.path.join(FIGURE_DIR, "overall_test_mse_vs_training_size.png"),
    )

    # Plot 2: Prediction MSE for Specific Timesteps (using the largest training size)
    if filtered_training_sizes_overall:
        largest_train_size = max(filtered_training_sizes_overall)
        if largest_train_size in filtered_mse_per_sequence:
            plot_prediction_mse_for_specific_timesteps(
                plot_sequence_indices,
                filtered_mse_per_sequence[largest_train_size],
                largest_train_size,
                save_path=os.path.join(
                    FIGURE_DIR,
                    f"prediction_mse_per_sequence_size_{largest_train_size}.png",
                ),
            )
        else:
            print(
                f"Could not generate Plot 2 for largest size {largest_train_size} as it might have been filtered out."
            )

    # Plot 3: Varying Training Size with MSE per Timestep
    plot_varying_training_size_with_mse_per_timestep(
        plot_sequence_indices,
        filtered_mse_per_sequence,
        save_path=os.path.join(
            FIGURE_DIR, "varying_training_size_mse_per_sequence.png"
        ),
    )

    print(f"\nPlots saved in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
