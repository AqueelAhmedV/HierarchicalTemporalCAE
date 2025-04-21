# --- main.py (Modular Comparison) ---

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import time
import json  # For saving results

# --- Local Imports ---
from plotting import plot_velocity_field  # Add other plotting imports if needed
from preprocess import prepare_for_autoencoder, timestep_data_generator
from models import (
    build_temporal_cae,
    build_hcae_temporal_subnetwork,  # Use the new temporal HCAE builder
    build_htcae_subnetwork,
)

# --- Configuration ---
# Data/General Config
CACHE_FILE = "./data/processed/temporal_data_new.npz"
CSV_FILE = "./data/csv/timestamp_data_full.csv"
MODEL_DIR = "./models/comparison/"
FIGURE_DIR = "./figures/comparison/"
RESULTS_FILE = "./results/comparison_results.json"
INPUT_SHAPE_SPATIAL = (256, 80, 2)
SEQUENCE_LENGTH = 4
TRAINING_SIZES = [80]  # [50, 80, 120] Add more sizes later

# Model Config (Common)
LATENT_DIM = 64
MODES_PER_FAMILY = 64  # For HCAE/HTCAE
N_FAMILIES = 4

# Training Config (Common)
BATCH_SIZE = 12
EPOCHS = 50  # Adjust as needed
PATIENCE_EARLY_STOPPING = 10
PATIENCE_LR_REDUCTION = 5
LOSS = "mse"

# --- Utility Functions ---


def get_optimizer():
    """Creates a new optimizer instance."""
    return keras.optimizers.Adamax()


def get_callbacks(checkpoint_path):
    """Creates standard callbacks for training."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=PATIENCE_LR_REDUCTION, verbose=1
        ),
        # keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_path,
        #     monitor="val_loss",
        #     save_best_only=True,
        #     save_weights_only=False,  # Save full model
        #     verbose=1,
        # ),
    ]


def prepare_sequences(data_array, sequence_length):
    """Helper to create overlapping sequences."""
    return np.array(
        [
            data_array[i : i + sequence_length]
            for i in range(len(data_array) - sequence_length + 1)
        ]
    )


# --- Training Functions ---


def train_standard_ae(model, train_seq, val_data, config, checkpoint_path):
    """Trains a standard (non-hierarchical) AE like TCAE."""
    best_model = None
    if os.path.exists(checkpoint_path):
        best_model = keras.models.load_model(
            checkpoint_path
        )  # Load the best saved model
    else:
        print(f"Training standard AE: {model.name}")
        start_time = time.time()
        optimizer = get_optimizer()
        model.compile(optimizer=optimizer, loss=config["loss"])
        callbacks = get_callbacks(checkpoint_path)

        history = model.fit(
            train_seq,
            train_seq,  # Target is itself
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_data=val_data if isinstance(val_data, tuple) else None,
            validation_split=val_data if isinstance(val_data, float) else None,
            callbacks=callbacks,
            verbose=1,
        )
        best_model = model
        best_model.save(checkpoint_path)
        # Save the model if no checkpointLoad the best saved model
        training_time = time.time() - start_time
        print(f"Standard AE training finished in {training_time:.2f} seconds.")
    return best_model, None, 0


def train_hierarchical_ae(
    build_subnetwork_func, original_train_seq, val_data, config, model_base_name
):
    """Trains a hierarchical AE (HCAE or HTCAE)."""
    print(f"Training hierarchical AE: {model_base_name}")
    start_time = time.time()
    build_args = config["build_args"]
    train_args = config["train_config"]
    n_families = train_args["n_families"]
    fit_args = {
        "epochs": train_args["epochs"],
        "batch_size": train_args["batch_size"],
        "validation_data": val_data if isinstance(val_data, tuple) else None,
        "validation_split": val_data if isinstance(val_data, float) else None,
        "verbose": 1,
    }

    current_input_sequences = original_train_seq
    cumulative_train_prediction = np.zeros_like(original_train_seq)
    trained_models = []
    all_histories = {}

    for i in range(n_families):
        print(f"\nTraining Subnetwork {i + 1}/{n_families} for {model_base_name}...")
        sub_model_name = f"{model_base_name}_sub_{i + 1}"
        checkpoint_path = os.path.join(
            train_args["model_dir"], f"{sub_model_name}_best.keras"
        )

        # Build subnetwork
        subnetwork = build_subnetwork_func(**build_args, name=sub_model_name)

        # Compile subnetwork with a new optimizer
        optimizer = get_optimizer()
        subnetwork.compile(optimizer=optimizer, loss=train_args["loss"])

        # Define Callbacks
        callbacks = get_callbacks(checkpoint_path)

        # Calculate input for this subnetwork
        input_for_current_fit = original_train_seq - cumulative_train_prediction

        # Train the subnetwork
        history = subnetwork.fit(
            input_for_current_fit,
            input_for_current_fit,
            callbacks=callbacks,
            **fit_args,
        )
        all_histories[sub_model_name] = history.history

        # Load the best saved model
        print(f"Loading best model from {checkpoint_path}")
        best_model = keras.models.load_model(checkpoint_path)

        # Predict contribution and update cumulative prediction
        print("Predicting contribution...")
        subnetwork_prediction = best_model.predict(
            input_for_current_fit, batch_size=config["batch_size"]
        )
        cumulative_train_prediction += subnetwork_prediction

        trained_models.append(best_model)

    training_time = time.time() - start_time
    print(f"Hierarchical training finished in {training_time:.2f} seconds.")
    # Return list of models, aggregated history, final train prediction, time
    return trained_models, all_histories, cumulative_train_prediction, training_time


# --- Prediction Functions ---


def predict_standard_ae(model, test_seq, config):
    """Predicts using a standard AE."""
    print(f"Predicting with standard AE: {model.name}")
    start_time = time.time()
    predictions = model.predict(test_seq, batch_size=config["batch_size"])
    prediction_time = time.time() - start_time
    print(f"Standard AE prediction finished in {prediction_time:.2f} seconds.")
    return predictions, prediction_time


def predict_hierarchical_ae(models_list, original_test_seq, config):
    """Predicts using a list of hierarchical AE models."""
    print(f"Predicting with hierarchical AE ({len(models_list)} subnetworks)...")
    start_time = time.time()
    cumulative_prediction = np.zeros_like(original_test_seq)

    for i, model in enumerate(models_list):
        # print(f"Predicting with subnetwork {i+1}/{len(models_list)}...") # Verbose
        current_input_for_model = original_test_seq - cumulative_prediction
        sub_prediction = model.predict(
            current_input_for_model, batch_size=config["batch_size"], verbose=0
        )
        cumulative_prediction += sub_prediction

    prediction_time = time.time() - start_time
    print(f"Hierarchical prediction finished in {prediction_time:.2f} seconds.")
    return cumulative_prediction, prediction_time


# --- Evaluation ---
def evaluate_model(original_seq, predicted_seq):
    """Calculates Mean Squared Error."""
    return np.mean((original_seq - predicted_seq) ** 2)


# --- Main Execution ---
def main():
    # --- Ensure Directories Exist ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # --- Load or Process Data ---
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        cached_data = np.load(CACHE_FILE)
        data_array = cached_data["data"]
        # timesteps = cached_data["timesteps"] # Timestamps not used in this version
        print(f"Loaded data shape: {data_array.shape}")
    else:
        # (Data processing logic remains the same as your last version)
        print(f"Processing data from {CSV_FILE}...")
        all_data, all_timesteps = [], []
        generator = timestep_data_generator(CSV_FILE)
        count = 0
        for timestep, data in generator:
            processed_data = prepare_for_autoencoder(data)
            all_data.append(processed_data[np.newaxis, ...])
            all_timesteps.append(timestep)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} timesteps...")
        if not all_data:
            print("Error: No data loaded.")
            return

        data_array = np.concatenate(all_data, axis=0)
        timesteps = np.array(all_timesteps)
        print("Sorting data...")
        sort_idx = np.argsort(timesteps)
        data_array = data_array[sort_idx]
        print(f"Final processed data shape: {data_array.shape}")
        print(f"Caching data to {CACHE_FILE}...")
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        np.savez(CACHE_FILE, data=data_array, timesteps=timesteps)
        print("Data cached.")

    # --- Define Model Configurations ---
    model_configs = {
        "tcae": {
            "build_func": build_temporal_cae,
            "train_func": train_standard_ae,
            "predict_func": predict_standard_ae,
            "is_hierarchical": False,
            "build_args": {  # Args for build_temporal_cae
                "input_shape_spatial": INPUT_SHAPE_SPATIAL,
                "sequence_length": SEQUENCE_LENGTH,
                "latent_dim": LATENT_DIM
                * N_FAMILIES,  # Standard TCAE uses total latent dim maybe? Or just LATENT_DIM? Let's try total. Adjust if needed.
                "use_skip_connections": True,
                # filters, etc. use defaults from models.py
            },
            "train_config": {  # Common training args + loss
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "loss": LOSS,
                "model_dir": MODEL_DIR,
            },
        },
        "hcae": {
            "build_func": build_hcae_temporal_subnetwork,  # The specific temporal builder
            "train_func": train_hierarchical_ae,
            "predict_func": predict_hierarchical_ae,
            "is_hierarchical": True,
            "build_args": {  # Args for build_hcae_temporal_subnetwork
                "input_shape_spatial": INPUT_SHAPE_SPATIAL,
                "sequence_length": SEQUENCE_LENGTH,
                "latent_dim": MODES_PER_FAMILY,
                "use_skip_connections": True,
                # filters use defaults (deeper) from that function in models.py
            },
            "train_config": {  # Common training args + hierarchy args + loss
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "n_families": N_FAMILIES,
                "loss": LOSS,
                "model_dir": MODEL_DIR,
            },
        },
        "htcae": {
            "build_func": build_htcae_subnetwork,  # The specific shallower builder
            "train_func": train_hierarchical_ae,
            "predict_func": predict_hierarchical_ae,
            "is_hierarchical": True,
            "build_args": {  # Args for build_htcae_subnetwork
                "input_shape_spatial": INPUT_SHAPE_SPATIAL,
                "sequence_length": SEQUENCE_LENGTH,
                "latent_dim": MODES_PER_FAMILY,
                "use_skip_connections": True,
                # filters use defaults (shallower) from that function in models.py
            },
            "train_config": {  # Common training args + hierarchy args + loss
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "n_families": N_FAMILIES,
                "loss": LOSS,
                "model_dir": MODEL_DIR,
            },
        },
    }

    all_results = {}

    # --- Loop over Training Sizes ---
    for train_size in TRAINING_SIZES:
        print(f"\n{'=' * 15} Processing for Training Size: {train_size} {'=' * 15}")
        results_key_base = f"train_size_{train_size}"
        all_results[results_key_base] = {}

        if train_size >= len(data_array) - SEQUENCE_LENGTH:
            print(f"Warning: train_size {train_size} too large. Skipping.")
            continue

        # --- Prepare Data Splits and Sequences ---
        val_split_index = max(SEQUENCE_LENGTH, int(train_size * 0.8))
        train_data_actual = data_array[:val_split_index]
        val_data = data_array[val_split_index:train_size]
        test_data = data_array[train_size:]

        print("Preparing sequences...")
        original_train_seq = prepare_sequences(train_data_actual, SEQUENCE_LENGTH)
        if len(val_data) >= SEQUENCE_LENGTH:
            validation_seq = prepare_sequences(val_data, SEQUENCE_LENGTH)
            val_data_for_fit = (validation_seq, validation_seq)
        else:
            print(
                "Warning: Not enough validation data for sequences, using validation_split."
            )
            val_data_for_fit = 0.2  # Use 20% of train data if val set too small

        test_seq = prepare_sequences(test_data, SEQUENCE_LENGTH)
        test_seq_eval = test_seq[:100]  # Subset for faster evaluation

        print(f"Train sequences: {original_train_seq.shape}")
        if isinstance(val_data_for_fit, tuple):
            print(f"Validation sequences: {val_data_for_fit[0].shape}")
        print(f"Test sequences (eval subset): {test_seq_eval.shape}")

        # --- Loop over Model Types ---
        for model_type, config in model_configs.items():
            print(f"\n------- Running Model Type: {model_type.upper()} -------")
            results_key_model = f"{results_key_base}_{model_type}"
            model_base_name = (
                f"{model_type}_train_{train_size}"  # Used for checkpoint naming
            )

            trained_model_or_list = None
            history_or_dict = None
            train_pred = None
            train_time = 0

            try:  # Add error handling per model
                if not config["is_hierarchical"]:
                    # --- Standard AE (TCAE) ---
                    model = config["build_func"](
                        **config["build_args"], name=model_base_name
                    )
                    checkpoint_path = os.path.join(
                        MODEL_DIR, f"{model_base_name}_best.keras"
                    )
                    trained_model_or_list, history_or_dict, train_time = (
                        train_standard_ae(
                            model,
                            original_train_seq,
                            val_data_for_fit,
                            config["train_config"],
                            checkpoint_path,
                        )
                    )
                    # Predict on training data for MSE calc
                    train_pred, _ = predict_standard_ae(
                        trained_model_or_list,
                        original_train_seq,
                        config["train_config"],
                    )

                else:
                    # --- Hierarchical AE (HCAE/HTCAE) ---
                    trained_model_or_list, history_or_dict, train_pred, train_time = (
                        train_hierarchical_ae(
                            config=config,
                            original_train_seq=original_train_seq,
                            # val_data_for_fit=val_data_for_fit,
                            val_data=val_data_for_fit,
                            model_base_name=model_base_name,
                            build_subnetwork_func=config["build_func"],
                        )
                    )

                # --- Prediction on Test Set ---
                if trained_model_or_list is not None:
                    if not config["is_hierarchical"]:
                        test_pred, pred_time = predict_standard_ae(
                            trained_model_or_list, test_seq_eval, config["train_config"]
                        )
                    else:
                        test_pred, pred_time = predict_hierarchical_ae(
                            trained_model_or_list, test_seq_eval, config["train_config"]
                        )

                    # --- Evaluation ---
                    train_mse = evaluate_model(original_train_seq, train_pred)
                    test_mse = evaluate_model(test_seq_eval, test_pred)

                    print(
                        f"\n** {model_type.upper()} Results (Train Size {train_size}) **"
                    )
                    print(f"Training Time: {train_time:.2f} s")
                    print(f"Prediction Time (Test): {pred_time:.2f} s")
                    print(f"Final Training MSE: {train_mse:.6f}")
                    print(f"Final Test MSE:     {test_mse:.6f}")

                    # Store results
                    all_results[results_key_model] = {
                        "train_size": train_size,
                        "model_type": model_type,
                        "train_mse": float(
                            train_mse
                        ),  # Convert numpy float -> python float
                        "test_mse": float(test_mse),
                        "train_time_s": round(train_time, 2),
                        "pred_time_s": round(pred_time, 2),
                        # Add history min val_loss?
                    }

                    # --- Plotting ---
                    plot_velocity_field(
                        original=test_seq_eval[0][
                            -1
                        ],  # Last frame of first test sequence
                        reconstructed=test_pred[0][-1],
                        title=f"{model_type.upper()} (Size {train_size}) - Test Recon (Seq 0)",
                        save_path=os.path.join(
                            FIGURE_DIR, f"{model_base_name}_test_comparison.png"
                        ),
                    )

                else:
                    print(
                        f"Model training failed for {model_type.upper()}, skipping prediction and evaluation."
                    )
                    all_results[results_key_model] = {
                        "status": "failed",
                        "train_size": train_size,
                        "model_type": model_type,
                    }

            except Exception as e:
                print(
                    f"!!!!!! ERROR processing model {model_type.upper()} for train size {train_size} !!!!!!"
                )
                print(f"Error: {e}")
                import traceback

                traceback.print_exc()
                all_results[results_key_model] = {
                    "status": "error",
                    "message": str(e),
                    "train_size": train_size,
                    "model_type": model_type,
                }

            print("-" * 40)

    # --- Save All Results ---
    print("\nSaving all results...")
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    # --- Print Summary Table ---
    print("\n--- Comparison Summary ---")
    print(
        f"{'Train Size':<12} | {'Model Type':<10} | {'Train MSE':<12} | {'Test MSE':<12} | {'Train Time (s)':<15} | {'Pred Time (s)':<15}"
    )
    print("-" * 80)
    for key, result in all_results.items():
        if result.get("status") in ["failed", "error"]:
            print(
                f"{result.get('train_size', 'N/A'):<12} | {result.get('model_type', 'N/A'):<10} | {'-':<12} | {'-':<12} | {result.get('status', 'N/A').upper()}"
            )
        else:
            print(
                f"{result['train_size']:<12} | {result['model_type']:<10} | {result['train_mse']:.6f} | {result['test_mse']:.6f} | {result['train_time_s']:<15.2f} | {result['pred_time_s']:<15.2f}"
            )
    print("-" * 80)


if __name__ == "__main__":
    main()
