import matplotlib.pyplot as plt
import numpy as np

def plot_velocity_field(original, reconstructed, title=None, save_path=None):
    """Plot original, reconstructed velocity fields, and error"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    x_range, y_range = 35.0, 20.0
    aspect_ratio = y_range / x_range
    
    # Ensure correct shape by removing any singleton dimensions
    original = np.squeeze(original)
    reconstructed = np.squeeze(reconstructed)
    
    # Calculate magnitudes
    magnitude_orig = np.sqrt(original[..., 0]**2 + original[..., 1]**2)
    magnitude_recon = np.sqrt(reconstructed[..., 0]**2 + reconstructed[..., 1]**2)
    error = np.abs(magnitude_orig - magnitude_recon)
    
    # Plot original
    im1 = ax1.imshow(magnitude_orig, cmap='viridis', aspect=aspect_ratio)
    ax1.set_title('Original' if not title else f'{title} - Original')
    plt.colorbar(im1, ax=ax1)
    
    # Plot reconstruction
    im2 = ax2.imshow(magnitude_recon, cmap='viridis', aspect=aspect_ratio)
    ax2.set_title('Reconstructed' if not title else f'{title} - Reconstructed')
    plt.colorbar(im2, ax=ax2)
    
    # Plot error
    im3 = ax3.imshow(error, cmap='hot', aspect=aspect_ratio)
    ax3.set_title('Error Distribution')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_training_history(history, title='Model Loss During Training'):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_hcae_history(histories):
    """Plot HCAE training history with subnetwork separation"""
    plt.figure(figsize=(10, 6))
    all_losses = []
    for history in histories:
        all_losses.extend(history.history['loss'])

    plt.plot(all_losses)
    plt.title('HCAE Training Loss (All Subnetworks)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.ylim(0, 1)

    # Add vertical lines to separate subnetworks
    cumulative_epochs = 0
    for i, history in enumerate(histories):
        cumulative_epochs += len(history.history['loss'])
        plt.axvline(x=cumulative_epochs, color='r', linestyle='--', alpha=0.5)
        plt.text(cumulative_epochs-len(history.history['loss'])/2, 0.9, 
                f'Subnetwork {i+1}', horizontalalignment='center')
    plt.show()

def compare_predictions(original, pred1, pred2, title1="Method 1", title2="Method 2", plot_type="error"):
    """
    Compare two different prediction methods
    
    Args:
        original: Original input data (shape: [..., height, width, channels])
        pred1: First prediction
        pred2: Second prediction
        title1: Name of first method
        title2: Name of second method
        plot_type: "error" for reconstruction error or "field" for velocity field
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    x_range, y_range = 35.0, 20.0
    aspect_ratio = y_range / x_range
    
    # Ensure arrays are 4D by adding singleton dimensions if needed
    if original.ndim == 3:
        original = original[np.newaxis, ...]
    if pred1.ndim == 3:
        pred1 = pred1[np.newaxis, ...]
    if pred2.ndim == 3:
        pred2 = pred2[np.newaxis, ...]
    
    if plot_type == "error":
        # Plot reconstruction error for method 1
        diff1 = np.sqrt((original[0, :, :, 0] - pred1[0, :, :, 0])**2 + 
                        (original[0, :, :, 1] - pred1[0, :, :, 1])**2)
        im1 = ax1.imshow(diff1, cmap='viridis', aspect=aspect_ratio)
        ax1.set_title(f'{title1}')
        plt.colorbar(im1, ax=ax1)
        
        # Plot reconstruction error for method 2
        diff2 = np.sqrt((original[0, :, :, 0] - pred2[0, :, :, 0])**2 + 
                        (original[0, :, :, 1] - pred2[0, :, :, 1])**2)
        im2 = ax2.imshow(diff2, cmap='viridis', aspect=aspect_ratio)
        ax2.set_title(f'{title2}')
        plt.colorbar(im2, ax=ax2)
        
    elif plot_type == "field":
        # Plot velocity field for method 1
        magnitude1 = np.sqrt(pred1[0, :, :, 0]**2 + pred1[0, :, :, 1]**2)
        im1 = ax1.imshow(magnitude1, cmap='viridis', aspect=aspect_ratio)
        ax1.set_title(f'{title1}')
        plt.colorbar(im1, ax=ax1)
        
        # Plot velocity field for method 2
        magnitude2 = np.sqrt(pred2[0, :, :, 0]**2 + pred2[0, :, :, 1]**2)
        im2 = ax2.imshow(magnitude2, cmap='viridis', aspect=aspect_ratio)
        ax2.set_title(f'{title2}')
        plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()

def plot_temporal_reconstructions(data_array, pred, timesteps, selected_timesteps=[0, 7, 14, 21]):
    """Plot velocity field reconstructions for selected timesteps"""
    for t in selected_timesteps:
        plot_velocity_field(
            data_array[t:t+1],  # Original
            pred[t:t+1]
        )

def plot_temporal_mse(data_array, pred, timesteps):
    """Plot MSE over time for temporal predictions"""
    timestep_mses = np.mean((data_array.squeeze(1) - pred.squeeze(1))**2, axis=(1,2,3))
    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, timestep_mses)
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title('Reconstruction Error Over Time')
    plt.grid(True)
    plt.show()

def plot_temporal_comparison(actual_data, predicted_data, timesteps, num_samples=4, title=""):
    """Plot comparison between actual and reconstructed data for selected timesteps"""
    # Calculate velocity magnitudes
    actual_mag = np.sqrt(actual_data[..., 0]**2 + actual_data[..., 1]**2)
    pred_mag = np.sqrt(predicted_data[..., 0]**2 + predicted_data[..., 1]**2)
    
    # Select evenly spaced timesteps
    step = len(timesteps) // num_samples
    selected_indices = range(0, len(timesteps), step)[:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 8))
    fig.suptitle('Actual vs Reconstructed Velocity Fields' + f' {title}')
    
    for i, idx in enumerate(selected_indices):
        # Plot actual data
        im1 = axes[0, i].imshow(actual_mag[idx, 0], aspect=0.3125, cmap='viridis')
        axes[0, i].set_title(f'Actual t={timesteps[idx]:.2f}')
        
        # Plot reconstructed data
        im2 = axes[1, i].imshow(pred_mag[idx, 0], aspect=0.3125, cmap='viridis')
        axes[1, i].set_title(f'Reconstructed t={timesteps[idx]:.2f}')
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[0, :].ravel().tolist(), label='Velocity Magnitude')
    plt.colorbar(im2, ax=axes[1, :].ravel().tolist(), label='Velocity Magnitude')
    
    # plt.tight_layout()
    plt.show()

def plot_temporal_error(actual_data, predicted_data, timesteps):
    """Plot MSE over time"""
    # Calculate MSE for each timestep
    mse_per_timestep = np.mean((actual_data - predicted_data)**2, axis=(1, 2, 3, 4))
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mse_per_timestep, 'b-', label='MSE')
    plt.xlabel('Time')
    plt.ylabel('Mean Squared Error')
    plt.title('Reconstruction Error Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

