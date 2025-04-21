from main1 import cae
import numpy as np


def interpolate_fields(input_data1, input_data2, num_steps=5):
    """
    Interpolate between two velocity fields in latent space
    """
    # Get latent representations
    latent1 = cae.encode(input_data1)
    latent2 = cae.encode(input_data2)

    # Create interpolations
    alphas = np.linspace(0, 1, num_steps)
    interpolated_fields = []

    for alpha in alphas:
        # Interpolate in latent space
        interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
        # Decode to get velocity field
        interpolated_field = cae.decode(interpolated_latent)
        interpolated_fields.append(interpolated_field)

    return interpolated_fields
