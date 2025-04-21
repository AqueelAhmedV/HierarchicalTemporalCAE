import os
import pydot.exceptions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
import keras
from keras.api.utils import plot_model




class ConvAutoEncoder:
    def __init__(self, input_shape, output_dim, filters=[32, 64, 128, 256],
                 kernel=(3,3), stride=(1,1), strideundo=2, pool=(2,2),
                 optimizer="adamax", lossfn="mse"):
        # For now, assuming input_shape is mxnxc, and m,n are multiples of 2.

        self.input_shape = input_shape
        self.output_dim  = output_dim

        # define encoder architecture
        self.encoder = keras.models.Sequential()
        self.encoder.add(keras.layers.InputLayer(input_shape))
        for i in range(len(filters)):
            self.encoder.add(keras.layers.Conv2D(filters=filters[i], kernel_size=kernel, strides=stride, activation='elu', padding='same'))
            self.encoder.add(keras.layers.MaxPooling2D(pool_size=pool))
        self.encoder.add(keras.layers.Flatten())
        self.encoder.add(keras.layers.Dense(output_dim))

        # define decoder architecture
        self.decoder = keras.models.Sequential()
        self.decoder.add(keras.layers.InputLayer((output_dim,)))
        self.decoder.add(keras.layers.Dense(filters[-1] * int(input_shape[0]/(2**len(filters))) * int(input_shape[1]/(2**len(filters)))))
        self.decoder.add(keras.layers.Reshape((int(input_shape[0]/(2**len(filters))), int(input_shape[1]/(2**len(filters))), filters[-1])))
        for i in range(1,len(filters)):
            self.decoder.add(keras.layers.Conv2DTranspose(filters=filters[len(filters)-i], kernel_size=kernel, strides=strideundo, activation='elu', padding='same'))
        self.decoder.add(    keras.layers.Conv2DTranspose(filters=input_shape[2],          kernel_size=kernel, strides=strideundo, activation=None,  padding='same'))

        # compile model
        input         = keras.layers.Input(input_shape)
        code          = self.encoder(input)
        reconstructed = self.decoder(code)

        self.ae = keras.models.Model(inputs=input, outputs=reconstructed)
        self.ae.compile(optimizer=optimizer, loss=lossfn)


    def fit(self, x, epochs=25, callbacks=[keras.callbacks.ProgbarLogger()], **kwargs):
        """
        Train the autoencoder
        
        Args:
            x: Input data
            epochs: Number of epochs to train
            callbacks: List of Keras callbacks
            **kwargs: Additional arguments to pass to model.fit
        
        Returns:
            History object containing training history
        """
        return self.ae.fit(x=x, y=x, epochs=epochs, callbacks=callbacks, **kwargs)


    def save_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.encoder.save_weights(os.path.join(path, prefix + "encoder.weights.h5"))
        self.decoder.save_weights(os.path.join(path, prefix + "decoder.weights.h5"))


    def load_weights(self, path=None, prefix=""):
        if path is None: path = os.getcwd()
        self.encoder.load_weights(os.path.join(path, prefix + "encoder.weights.h5"))
        self.decoder.load_weights(os.path.join(path, prefix + "decoder.weights.h5"))


    def encode(self, input):
        return self.encoder.predict(input)


    def decode(self, codes):
        return self.decoder.predict(codes)

    def predict(self, input):
        return self.encode(input)


class HierarchicalCAE:
    def __init__(self, input_shape, n_families=4, modes_per_family=72,
                 filters=[16, 32, 64, 128], kernel=(3,3), stride=(1,1), 
                 pool=(2,2), optimizer="adamax", lossfn="mse"):
        """
        Hierarchical Convolutional Autoencoder
        
        Args:
            input_shape: Shape of input data (height, width, channels)
            n_families: Number of mode families (subnetworks)
            modes_per_family: Number of modes per family
            filters: List of filter sizes for conv layers
            kernel: Kernel size for conv layers
            stride: Stride for conv layers
            pool: Pooling size
        """
        self.input_shape = input_shape
        self.n_families = n_families
        self.modes_per_family = modes_per_family
        self.subnetworks = []
        
        # Create hierarchical subnetworks
        for i in range(n_families):
            # Each subnetwork is a CAE
            cae = ConvAutoEncoder(
                input_shape=input_shape,
                output_dim=modes_per_family,
                filters=filters,
                kernel=kernel,
                stride=stride,
                pool=pool,
                optimizer=optimizer,
                lossfn=lossfn
            )
            self.subnetworks.append(cae)
    
    def fit(self, x, epochs=100, batch_size=32, callbacks=None):
        """
        Train the hierarchical autoencoder
        Following equations (3)-(6) from the paper
        """
        residual = x.copy()
        histories = []
        
        # Train each subnetwork hierarchically
        for i in range(self.n_families):
            print(f"\nTraining subnetwork {i+1}/{self.n_families}")
            
            # Train current subnetwork on residual
            history = self.subnetworks[i].fit(
                residual, 
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            histories.append(history)
            
            # Update residual for next subnetwork
            # residual = x - sum(reconstructions from trained subnetworks)
            reconstruction = self.subnetworks[i].ae.predict(residual)
            residual = residual - reconstruction
            
            print(f"Residual MSE after subnetwork {i+1}: {np.mean(residual**2):.10f}")
        
        return histories
    
    def predict(self, x):
        """
        Reconstruct input using all subnetworks
        """
        reconstruction = np.zeros_like(x)
        residual = x.copy()
        
        for i in range(self.n_families):
            current_reconstruction = self.subnetworks[i].ae.predict(residual)
            reconstruction += current_reconstruction
            residual = residual - current_reconstruction
            
        return reconstruction
    
    def encode(self, x):
        """
        Get latent representations from all subnetworks
        """
        codes = []
        residual = x.copy()
        
        for i in range(self.n_families):
            code = self.subnetworks[i].encode(residual)
            codes.append(code)
            reconstruction = self.subnetworks[i].decode(code)
            residual = residual - reconstruction
            
        return np.concatenate(codes, axis=1)
    
    def save_weights(self, path="weights", prefix=""):
        """Save weights for all subnetworks"""
        for i, cae in enumerate(self.subnetworks):
            cae.save_weights(path=path, prefix=f"{prefix}subnetwork_{i+1}_")
    
    def load_weights(self, path="weights", prefix=""):
        """Load weights for all subnetworks"""
        for i, cae in enumerate(self.subnetworks):
            cae.load_weights(path=path, prefix=f"{prefix}subnetwork_{i+1}_")


class TemporalCAE:
    def __init__(self, input_shape, output_dim, filters=[32, 64, 128, 256],
                 kernel=(3,3), stride=(1,1), pool=(2,2),
                 optimizer="adamax", lossfn="mse"):
        self.input_shape = input_shape  # (timesteps, height, width, channels)
        self.output_dim = output_dim
        self.timesteps = input_shape[0]

        # Calculate total pooling factor
        self.total_pool_h = 1
        self.total_pool_w = 1
        
        # Spatial Encoder (CNN)
        self.spatial_encoder = keras.models.Sequential()
        self.spatial_encoder.add(keras.layers.TimeDistributed(
            keras.layers.Conv2D(filters=filters[0], kernel_size=kernel, 
                              strides=stride, activation='elu', padding='same'),
            input_shape=input_shape
        ))
        
        # Add remaining spatial encoding layers
        for f in filters[1:]:
            self.spatial_encoder.add(keras.layers.TimeDistributed(
                keras.layers.Conv2D(filters=f, kernel_size=kernel, 
                                  strides=stride, activation='elu', padding='same')
            ))
            self.spatial_encoder.add(keras.layers.TimeDistributed(
                keras.layers.MaxPooling2D(pool_size=pool)
            ))
            self.total_pool_h *= pool[0]
            self.total_pool_w *= pool[1]
        
        self.spatial_encoder.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        self.spatial_encoder.add(keras.layers.TimeDistributed(
            keras.layers.Dense(output_dim)
        ))

        # Temporal Processing (LSTM in latent space)
        self.temporal_processor = keras.models.Sequential()
        self.temporal_processor.add(keras.layers.LSTM(
            units=output_dim,
            return_sequences=True,
            input_shape=(self.timesteps, output_dim)
        ))
        
        # Spatial Decoder
        self.spatial_decoder = keras.models.Sequential()
        
        # Calculate spatial dimensions after pooling
        h = input_shape[1] // self.total_pool_h
        w = input_shape[2] // self.total_pool_w
        
        self.spatial_decoder.add(keras.layers.TimeDistributed(
            keras.layers.Dense(filters[-1] * h * w),
            input_shape=(self.timesteps, output_dim)
        ))
        self.spatial_decoder.add(keras.layers.TimeDistributed(
            keras.layers.Reshape((h, w, filters[-1]))
        ))

        # Upsampling layers
        for i in range(len(filters)-1, 0, -1):
            self.spatial_decoder.add(keras.layers.TimeDistributed(
                keras.layers.Conv2DTranspose(
                    filters=filters[i-1],
                    kernel_size=kernel,
                    strides=pool,
                    padding='same',
                    activation='elu'
                )
            ))

        # Final layer
        self.spatial_decoder.add(keras.layers.TimeDistributed(
            keras.layers.Conv2DTranspose(
                filters=input_shape[-1],
                kernel_size=kernel,
                strides=(1, 1),
                padding='same',
                activation=None
            )
        ))

        # Build full model
        input_layer = keras.layers.Input(input_shape)
        # Spatial encoding
        spatial_features = self.spatial_encoder(input_layer)
        # Temporal processing in latent space
        temporal_features = self.temporal_processor(spatial_features)
        # Spatial decoding
        reconstructed = self.spatial_decoder(temporal_features)

        self.ae = keras.models.Model(inputs=input_layer, outputs=reconstructed)
        self.ae.compile(optimizer=optimizer, loss=lossfn)

    def fit(self, x, epochs=25, callbacks=[keras.callbacks.ProgbarLogger()], **kwargs):
        """Train the autoencoder"""
        return self.ae.fit(x=x, y=x, epochs=epochs, callbacks=callbacks, **kwargs)

    def save_weights(self, path=None, prefix=""):
        """Save model weights"""
        if path is None: path = os.getcwd()
        self.spatial_encoder.save_weights(os.path.join(path, prefix + "spatial_encoder.weights.h5"))
        self.temporal_processor.save_weights(os.path.join(path, prefix + "temporal_processor.weights.h5"))
        self.spatial_decoder.save_weights(os.path.join(path, prefix + "spatial_decoder.weights.h5"))

    def load_weights(self, path=None, prefix=""):
        """Load model weights"""
        if path is None: path = os.getcwd()
        self.spatial_encoder.load_weights(os.path.join(path, prefix + "spatial_encoder.weights.h5"))
        self.temporal_processor.load_weights(os.path.join(path, prefix + "temporal_processor.weights.h5"))
        self.spatial_decoder.load_weights(os.path.join(path, prefix + "spatial_decoder.weights.h5"))

    def encode(self, input):
        """Get latent space representation"""
        spatial_features = self.spatial_encoder.predict(input)
        return self.temporal_processor.predict(spatial_features)

    def decode(self, codes):
        """Decode from latent space"""
        return self.spatial_decoder.predict(codes)

    def predict(self, input):
        """Full forward pass"""
        return self.ae.predict(input)


class HTemporalCAE:
    def __init__(self, input_shape, n_families=4, modes_per_family=64,
                 filters=[16, 32, 64], kernel=(3,3), stride=(1,1), pool=(2,2),
                 optimizer="adamax", lossfn="mse"):
        """
        Hierarchical Temporal Convolutional Autoencoder
        
        Args:
            input_shape: Shape of input data (timesteps, height, width, channels)
            n_families: Number of mode families (subnetworks)
            modes_per_family: Number of modes per family (latent dimension)
            filters: List of filter sizes for conv layers
            kernel: Kernel size for conv layers
            stride: Stride for conv layers
            pool: Pooling size
        """
        self.input_shape = input_shape
        self.n_families = n_families
        self.modes_per_family = modes_per_family
        self.subnetworks = []
        
        # Create hierarchical subnetworks
        for i in range(n_families):
            # Each subnetwork is a TemporalCAE
            temporal_cae = TemporalCAE(
                input_shape=input_shape,
                output_dim=modes_per_family,
                filters=filters,
                kernel=kernel,
                stride=stride,
                pool=pool,
                optimizer=optimizer,
                lossfn=lossfn
            )
            self.subnetworks.append(temporal_cae)
    
    def fit(self, x, epochs=100, batch_size=4, callbacks=None):
        """
        Train the hierarchical temporal autoencoder
        Following Fukami's hierarchical training approach with temporal consideration
        """
        residual = x.copy()
        histories = []
        
        # Train each subnetwork hierarchically
        for i in range(self.n_families):
            print(f"\nTraining temporal subnetwork {i+1}/{self.n_families}")
            
            # Train current subnetwork on residual
            history = self.subnetworks[i].fit(
                residual, 
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            histories.append(history)
            
            # Update residual for next subnetwork
            # residual = x - sum(reconstructions from trained subnetworks)
            reconstruction = self.subnetworks[i].predict(residual)
            residual = residual - reconstruction
            
            print(f"Residual MSE after temporal subnetwork {i+1}: {np.mean(residual**2):.10f}")
        
        return histories
    
    def predict(self, x):
        """
        Reconstruct input using all temporal subnetworks
        """
        reconstruction = np.zeros_like(x)
        residual = x.copy()
        
        for i in range(self.n_families):
            current_reconstruction = self.subnetworks[i].predict(residual)
            reconstruction += current_reconstruction
            residual = residual - current_reconstruction
            
        return reconstruction
    
    def encode(self, x):
        """
        Get latent representations from all temporal subnetworks
        Returns concatenated temporal-spatial codes
        """
        codes = []
        residual = x.copy()
        
        for i in range(self.n_families):
            code = self.subnetworks[i].encode(residual)
            codes.append(code)
            reconstruction = self.subnetworks[i].decode(code)
            residual = residual - reconstruction
            
        return np.concatenate(codes, axis=2)  # Concatenate along feature dimension
    
    def save_weights(self, path="weights", prefix=""):
        """Save weights for all temporal subnetworks"""
        for i, tcae in enumerate(self.subnetworks):
            tcae.save_weights(path=path, prefix=f"{prefix}temporal_subnetwork_{i+1}_")
    
    def load_weights(self, path="weights", prefix=""):
        """Load weights for all temporal subnetworks"""
        for i, tcae in enumerate(self.subnetworks):
            tcae.load_weights(path=path, prefix=f"{prefix}temporal_subnetwork_{i+1}_")

def export_model_architecture(model, model_name="model"):
    """
    Export model architecture in multiple formats
    
    Args:
        model: Keras model
        model_name: Name for the output files
    """
    # 1. Text summary to file
    with open(f'./img/{model_name}_summary.txt', 'w', encoding='utf-8') as f:
        # Redirect summary to file
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # 2. Visual representation
    try:
        plot_model(
            model,
            to_file=f'./img/{model_name}_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
            expand_nested=True
        )
    except Exception as e:
        print(f"Warning: Could not generate model plot. Error: {str(e)}")
        print("Make sure you have graphviz installed on your system.")
    
    # 3. TensorBoard (optional)
    import datetime
    log_dir = "logs/model_architecture/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    return tensorboard_callback

# Example usage for your models:
def export_all_architectures(input_shape):
    """Export architectures for all models"""
    # CAE
    cae = ConvAutoEncoder(
        input_shape=input_shape,
        output_dim=64,
        filters=[16, 32, 64, 128]
    )
    export_model_architecture(cae.ae, "cae")
    
    # HCAE
    hcae = HierarchicalCAE(
        input_shape=input_shape,
        n_families=4,
        modes_per_family=72
    )
    # Export first subnetwork as example
    export_model_architecture(hcae.subnetworks[0].ae, "hcae_subnetwork")
    
    # Temporal CAE
    temporal_shape = (4, *input_shape)  # Add sequence length
    tcae = TemporalCAE(
        input_shape=temporal_shape,
        output_dim=64,
        filters=[16, 32, 64]
    )
    export_model_architecture(tcae.ae, "temporal_cae")
    
    # HTCAE
    htcae = HTemporalCAE(
        input_shape=temporal_shape,
        n_families=4,
        modes_per_family=64
    )
    # Export first subnetwork as example
    export_model_architecture(htcae.subnetworks[0].ae, "htcae_subnetwork")

# Call with your input shape
input_shape = (256, 80, 2)
export_all_architectures(input_shape)