import tensorflow as tf
import keras
from keras import layers
import math  # Import math for calculating product

# --- Configuration Defaults ---
DEFAULT_INPUT_SHAPE_SPATIAL = (256, 80, 2)  # H, W, C (u, v)
DEFAULT_LATENT_DIM = 64
DEFAULT_FILTERS_CAE = [16, 32, 64, 128]
DEFAULT_FILTERS_HTCAE = [16, 32, 64]  # Shallower for HTCAE (Sec 3.3.4)
DEFAULT_KERNEL_SIZE = (3, 3)
DEFAULT_POOL_SIZE = (2, 2)
DEFAULT_ACTIVATION = "elu"  # As per Sec 3.6.2
DEFAULT_SEQ_LENGTH = 4  # As per Sec 3.2 / 3.3.3

# --- Base CAE Blocks (Corrected) ---


def build_encoder(
    input_shape=DEFAULT_INPUT_SHAPE_SPATIAL,
    filters=DEFAULT_FILTERS_CAE,
    kernel_size=DEFAULT_KERNEL_SIZE,
    pool_size=DEFAULT_POOL_SIZE,
    activation=DEFAULT_ACTIVATION,
    latent_dim=DEFAULT_LATENT_DIM,
    name="encoder",
):
    """Builds the spatial encoder block."""
    inputs = keras.Input(shape=input_shape, name=f"{name}_input")
    x = inputs
    shape_before_flattening = None
    skip_connection_layers = {}  # Store outputs of POOL layers for skips

    # Convolutional layers with MaxPooling
    for i, f in enumerate(filters):
        layer_name_conv = f"{name}_conv_{i + 1}"
        layer_name_pool = f"{name}_pool_{i + 1}"
        x = layers.Conv2D(
            f, kernel_size, activation=activation, padding="same", name=layer_name_conv
        )(x)
        # Don't store conv outputs for this skip logic, store pool outputs
        # skip_connection_layers[layer_name_conv] = x
        if i < len(filters) - 1:
            x = layers.MaxPooling2D(pool_size, padding="same", name=layer_name_pool)(x)
            skip_connection_layers[layer_name_pool] = x  # Store POOLED output for skip

    # Shape before flattening for decoder
    shape_list = list(x.shape)
    shape_before_flattening = tuple(shape_list[1:])
    if None in shape_before_flattening:
        raise ValueError(
            f"Could not determine static shape before flattening in encoder. Got: {shape_before_flattening}"
        )

    # Flatten and Dense to latent space
    x = layers.Flatten(name=f"{name}_flatten")(x)
    latent_vector = layers.Dense(
        latent_dim, activation=activation, name=f"{name}_latent_dense"
    )(x)

    # Return model, shape tuple, and skips dictionary (containing pool outputs)
    encoder = keras.Model(inputs, [latent_vector, skip_connection_layers], name=name)
    return encoder, shape_before_flattening


def build_decoder(
    latent_dim=DEFAULT_LATENT_DIM,
    reshape_target_shape=None,
    filters=DEFAULT_FILTERS_CAE,
    kernel_size=DEFAULT_KERNEL_SIZE,
    activation=DEFAULT_ACTIVATION,
    output_channels=2,
    name="decoder",
):
    """Builds the spatial decoder block (no skip logic here)."""
    if reshape_target_shape is None:
        raise ValueError("reshape_target_shape must be provided from the encoder.")
    if not isinstance(reshape_target_shape, tuple) or not all(
        isinstance(d, int) for d in reshape_target_shape
    ):
        raise TypeError(
            f"reshape_target_shape must be a tuple of integers. Got: {reshape_target_shape}"
        )

    latent_inputs = keras.Input(shape=(latent_dim,), name=f"{name}_latent_input")

    reshape_units = math.prod(reshape_target_shape)
    x = layers.Dense(reshape_units, activation=activation, name=f"{name}_expand_dense")(
        latent_inputs
    )
    x = layers.Reshape(reshape_target_shape, name=f"{name}_reshape")(x)

    reversed_filters = filters[::-1]
    for i, f in enumerate(reversed_filters):
        use_strides = i > 0
        x = layers.Conv2DTranspose(
            filters=reversed_filters[i + 1]
            if i < len(reversed_filters) - 1
            else output_channels,
            kernel_size=kernel_size,
            strides=(2, 2) if use_strides else (1, 1),
            activation=activation if i < len(reversed_filters) - 1 else "linear",
            padding="same",
            name=f"{name}_tconv_{i + 1}",
        )(x)

    decoder_outputs = x
    decoder = keras.Model(latent_inputs, decoder_outputs, name=name)
    return decoder


# --- Convolutional Autoencoder (CAE) ---


def build_cae(
    input_shape=DEFAULT_INPUT_SHAPE_SPATIAL,
    filters=DEFAULT_FILTERS_CAE,
    kernel_size=DEFAULT_KERNEL_SIZE,
    pool_size=DEFAULT_POOL_SIZE,
    activation=DEFAULT_ACTIVATION,
    latent_dim=DEFAULT_LATENT_DIM,
    name="cae",
):
    """Builds the standard Convolutional Autoencoder."""
    encoder_model, shape_before_flattening = build_encoder(
        input_shape,
        filters,
        kernel_size,
        pool_size,
        activation,
        latent_dim,
        f"{name}_encoder",
    )
    decoder = build_decoder(
        latent_dim,
        shape_before_flattening,
        filters,
        kernel_size,
        activation,
        input_shape[-1],
        f"{name}_decoder",
    )

    input_layer = keras.Input(shape=input_shape, name=f"{name}_input")
    latent, _ = encoder_model(input_layer)  # Skips not used in base CAE
    reconstruction = decoder(latent)

    cae = keras.Model(input_layer, reconstruction, name=name)
    return cae


# --- Hierarchical CAE (HCAE) ---


def build_hcae_subnetwork(
    input_shape=DEFAULT_INPUT_SHAPE_SPATIAL,
    latent_dim=DEFAULT_LATENT_DIM,
    filters=DEFAULT_FILTERS_CAE,
    kernel_size=DEFAULT_KERNEL_SIZE,
    pool_size=DEFAULT_POOL_SIZE,
    activation=DEFAULT_ACTIVATION,
    name="hcae_subnetwork",
):
    """Builds a single subnetwork for HCAE (identical to base CAE)."""
    return build_cae(
        input_shape, filters, kernel_size, pool_size, activation, latent_dim, name
    )


print("--- HCAE Information ---")
print(
    "HCAE is typically implemented by training multiple CAE subnetworks sequentially."
)
print("Each subnetwork trains on the residual error of the previous ones.")
print("The build_hcae_subnetwork function provides the building block.")


# --- Temporal CAE Blocks (Corrected Skip Storage) ---


def build_temporal_encoder(
    input_shape=(DEFAULT_SEQ_LENGTH,) + DEFAULT_INPUT_SHAPE_SPATIAL,
    filters=DEFAULT_FILTERS_CAE,
    kernel_size=DEFAULT_KERNEL_SIZE,
    pool_size=DEFAULT_POOL_SIZE,
    activation=DEFAULT_ACTIVATION,
    latent_dim=DEFAULT_LATENT_DIM,
    name="temporal_encoder",
):
    """Builds the TimeDistributed spatial encoder with skip connection outputs."""
    inputs = keras.Input(shape=input_shape, name=f"{name}_input")
    x = inputs
    skip_connection_layers = {}  # Store POOL outputs

    # TimeDistributed Convolutional layers with MaxPooling
    for i, f in enumerate(filters):
        layer_name_conv = f"{name}_conv_{i + 1}"
        layer_name_pool = f"{name}_pool_{i + 1}"
        x = layers.TimeDistributed(
            layers.Conv2D(f, kernel_size, activation=activation, padding="same"),
            name=layer_name_conv,
        )(x)
        if i < len(filters) - 1:
            x = layers.TimeDistributed(
                layers.MaxPooling2D(pool_size, padding="same"), name=layer_name_pool
            )(x)
            skip_connection_layers[layer_name_pool] = x  # Store POOLED output

    # Shape before flattening (per timestep)
    shape_list = list(x.shape)
    shape_before_flattening = tuple(shape_list[2:])
    if None in shape_before_flattening:
        raise ValueError(
            f"Could not determine static shape before flattening in temporal encoder. Got: {shape_before_flattening}"
        )

    # Flatten and Dense per timestep
    x = layers.TimeDistributed(layers.Flatten(), name=f"{name}_flatten")(x)
    encoded_sequence = layers.TimeDistributed(
        layers.Dense(latent_dim, activation=activation), name=f"{name}_latent_dense_seq"
    )(x)

    # Return model, shape tuple, and skips dictionary (containing pool outputs)
    encoder = keras.Model(inputs, [encoded_sequence, skip_connection_layers], name=name)
    return encoder, shape_before_flattening


# --- Temporal Convolutional Autoencoder (TCAE) (Corrected Skip Logic) ---


def build_temporal_cae(
    input_shape_spatial=DEFAULT_INPUT_SHAPE_SPATIAL,
    sequence_length=DEFAULT_SEQ_LENGTH,
    filters=DEFAULT_FILTERS_CAE,
    kernel_size=DEFAULT_KERNEL_SIZE,
    pool_size=DEFAULT_POOL_SIZE,
    activation=DEFAULT_ACTIVATION,
    latent_dim=DEFAULT_LATENT_DIM,
    lstm_units=None,  # Defaults to latent_dim if None
    use_skip_connections=True,
    name="temporal_cae",
):
    """Builds the Temporal Convolutional Autoencoder with corrected skips."""
    input_shape_full = (sequence_length,) + input_shape_spatial
    if lstm_units is None:
        lstm_units = latent_dim

    # Build Encoder
    encoder_model, shape_before_flattening = build_temporal_encoder(
        input_shape_full,
        filters,
        kernel_size,
        pool_size,
        activation,
        latent_dim,
        f"{name}_encoder",
    )
    encoder_input = encoder_model.input
    encoded_sequence, skip_layers_dict = encoder_model(encoder_input)  # Unpack outputs

    # Build LSTM
    lstm_output = layers.LSTM(lstm_units, return_sequences=True, name=f"{name}_lstm")(
        encoded_sequence
    )

    # --- Reconstruct Decoder Graph Manually with Corrected Skip Logic ---
    x_dec = lstm_output

    if not isinstance(shape_before_flattening, tuple) or not all(
        isinstance(d, int) for d in shape_before_flattening
    ):
        raise TypeError(
            f"shape_before_flattening must be a tuple of integers. Got: {shape_before_flattening}"
        )
    reshape_units = math.prod(shape_before_flattening)

    x_dec = layers.TimeDistributed(
        layers.Dense(reshape_units, activation=activation),
        name=f"{name}_decoder_expand_dense_seq",
    )(x_dec)
    x_dec = layers.TimeDistributed(
        layers.Reshape(shape_before_flattening), name=f"{name}_decoder_reshape_seq"
    )(x_dec)  # Shape: e.g., (B, 4, 32, 10, 128)

    reversed_filters = filters[::-1]  # [128, 64, 32, 16]
    skips_for_decoder = skip_layers_dict if use_skip_connections else {}

    # Decoder loop: i=0..3 ; f=128, 64, 32, 16
    for i, f_enc in enumerate(reversed_filters):
        # Determine if this TConv layer corresponds to reversing a pooling operation
        # Pooling happens after Conv layers 1, 2, 3 (indices 0, 1, 2)
        # TConv layers 2, 3, 4 (indices i=1, 2, 3) reverse these poolings
        use_strides = i > 0
        layer_name = f"{name}_decoder_tconv_{i + 1}"

        # --- Add skip connection *before* Conv2DTranspose ---
        # Identify the corresponding encoder pool layer that this TConv reverses
        if use_strides:
            # TConv i=1 reverses Pool_3 (encoder idx 2)
            # TConv i=2 reverses Pool_2 (encoder idx 1)
            # TConv i=3 reverses Pool_1 (encoder idx 0)
            encoder_pool_index = (
                len(filters) - 1 - i
            )  # Index of the pool layer in encoder (e.g., 2 for i=1)
            skip_key = (
                f"{name}_encoder_pool_{encoder_pool_index + 1}"  # Key for pooled output
            )

            if use_skip_connections and skip_key in skips_for_decoder:
                skip_tensor = skips_for_decoder[skip_key]
                if skip_tensor is not None:
                    print(
                        f"Decoder loop {i}: Adding skip from {skip_key} (shape {skip_tensor.shape}) to input of {layer_name} (shape {x_dec.shape})"
                    )
                    # Check shapes before adding
                    if x_dec.shape[1:] == skip_tensor.shape[1:]:
                        x_dec = layers.Add(
                            name=f"{name}_decoder_add_skip_pool_{encoder_pool_index + 1}"
                        )([x_dec, skip_tensor])
                    else:
                        print(
                            f"Warning: Shape mismatch before TConv for skip {skip_key}. Decoder: {x_dec.shape}, Skip: {skip_tensor.shape}. Skipping Add."
                        )
        # --- End skip connection Add ---

        # --- Apply Conv2DTranspose ---
        # Determine the number of output filters for this layer
        num_filters_tconv = (
            reversed_filters[i + 1]
            if i < len(reversed_filters) - 1
            else input_shape_spatial[-1]
        )
        # Determine activation for this layer
        activation_tconv = activation if i < len(reversed_filters) - 1 else "linear"

        x_dec = layers.TimeDistributed(
            layers.Conv2DTranspose(
                filters=num_filters_tconv,
                kernel_size=kernel_size,
                strides=(2, 2) if use_strides else (1, 1),  # Upsample if reversing pool
                activation=activation_tconv,  # Apply activation within layer
                padding="same",
            ),
            name=layer_name,
        )(x_dec)
        # --- End Conv2DTranspose ---

    reconstruction = x_dec
    # --- End Decoder Graph ---

    # Create the final model
    tcae = keras.Model(encoder_input, reconstruction, name=name)
    return tcae


# --- Hierarchical Temporal CAE (HTCAE) ---


def build_htcae_subnetwork(
    input_shape_spatial=DEFAULT_INPUT_SHAPE_SPATIAL,
    sequence_length=DEFAULT_SEQ_LENGTH,
    latent_dim=DEFAULT_LATENT_DIM,
    filters=DEFAULT_FILTERS_HTCAE,  # Using shallower filters
    kernel_size=DEFAULT_KERNEL_SIZE,
    pool_size=DEFAULT_POOL_SIZE,
    activation=DEFAULT_ACTIVATION,
    lstm_units=None,
    use_skip_connections=True,
    name="htcae_subnetwork",
):
    """Builds a single temporal subnetwork for HTCAE (shallower filters)."""
    # NOTE: If HTCAE filters are different, the skip connection shapes
    # might not match if skips come from a deeper base CAE.
    # This implementation assumes skips are internal to the HTCAE subnetwork build.
    return build_temporal_cae(
        input_shape_spatial,
        sequence_length,
        filters,
        kernel_size,
        pool_size,
        activation,
        latent_dim,
        lstm_units,
        use_skip_connections,
        name,
    )


# In models.py, add this function:
def build_hcae_temporal_subnetwork(
    input_shape_spatial=DEFAULT_INPUT_SHAPE_SPATIAL,
    sequence_length=DEFAULT_SEQ_LENGTH,
    latent_dim=DEFAULT_LATENT_DIM,
    filters=DEFAULT_FILTERS_CAE,  # Use DEEPER filters for HCAE
    kernel_size=DEFAULT_KERNEL_SIZE,
    pool_size=DEFAULT_POOL_SIZE,
    activation=DEFAULT_ACTIVATION,
    lstm_units=None,
    use_skip_connections=True,  # Assuming HCAE also benefits from skips
    name="hcae_temporal_subnetwork",
):
    """Builds a single TEMPORAL subnetwork for HCAE comparison (uses deeper filters)."""
    # Reuses the TCAE builder but with the deeper CAE filter config
    return build_temporal_cae(
        input_shape_spatial,
        sequence_length,
        filters,
        kernel_size,
        pool_size,
        activation,
        latent_dim,
        lstm_units,
        use_skip_connections,
        name,
    )


# # --- Example Usage (Corrected) ---
# if __name__ == "__main__":
#     print("\n--- Building Example Models (Corrected) ---")

#     print("\nBuilding CAE...")
#     cae_model = build_cae()
#     cae_model.summary(expand_nested=True)
#     print(f"CAE Input shape: {cae_model.input_shape}")
#     print(f"CAE Output shape: {cae_model.output_shape}")
#     print("-" * 30)

#     hcae_sub = build_hcae_subnetwork()

#     print("\nBuilding Temporal CAE (with skips)...")
#     tcae_model_skips = build_temporal_cae(
#         use_skip_connections=True, name="tcae_with_skips"
#     )
#     tcae_model_skips.summary(expand_nested=True)
#     print(f"Temporal CAE (Skips) Input shape: {tcae_model_skips.input_shape}")
#     print(f"Temporal CAE (Skips) Output shape: {tcae_model_skips.output_shape}")
#     print("-" * 30)

#     print("\nBuilding Temporal CAE (no skips)...")
#     tcae_model_no_skips = build_temporal_cae(
#         use_skip_connections=False, name="tcae_no_skips"
#     )
#     tcae_model_no_skips.summary(expand_nested=True)
#     print(f"Temporal CAE (No Skips) Input shape: {tcae_model_no_skips.input_shape}")
#     print(f"Temporal CAE (No Skips) Output shape: {tcae_model_no_skips.output_shape}")
#     print("-" * 30)

#     print("\nBuilding HTCAE Subnetwork...")
#     htcae_sub = build_htcae_subnetwork()
#     # htcae_sub.summary(expand_nested=True) # Uncomment to see summary
#     print(f"HTCAE Subnetwork Input shape: {htcae_sub.input_shape}")
#     print(f"HTCAE Subnetwork Output shape: {htcae_sub.output_shape}")
#     print("-" * 30)

#     print("\n--- Model Build Complete ---")
