# models/autoencoder/autoencoder.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class CabinetDecisionAutoencoder:
    """Autoencoder for cabinet decisions analysis"""
    
    def __init__(self, input_dim=5000, encoding_dim=128):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.history = None
        self.scaler = MinMaxScaler()
        
    def build_model(self):
        """Build autoencoder architecture"""
        print(f"🏗️ Building autoencoder: {self.input_dim} → {self.encoding_dim} → {self.input_dim}")
        
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoding layers with dropout for regularization
        encoded = layers.Dense(2048, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(1024, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(512, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(256, activation='relu')(encoded)
        
        # Latent space
        latent = layers.Dense(self.encoding_dim, activation='relu', name='latent')(encoded)
        
        # Decoder
        decoded = layers.Dense(256, activation='relu')(latent)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(512, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(1024, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(2048, activation='relu')(decoded)
        
        # Output layer
        output_layer = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Models
        self.autoencoder = Model(input_layer, output_layer, name='autoencoder')
        self.encoder = Model(input_layer, latent, name='encoder')
        
        # Decoder model (for reconstruction from latent space)
        latent_input = layers.Input(shape=(self.encoding_dim,))
        decoder_layers = self.autoencoder.layers[-8:]  # Get last 8 layers (decoder part)
        decoder_output = latent_input
        for layer in decoder_layers:
            decoder_output = layer(decoder_output)
        self.decoder = Model(latent_input, decoder_output, name='decoder')
        
        # Compile
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Model built successfully")
        self.autoencoder.summary()
        
        return self.autoencoder
    
    def prepare_data(self, X_sparse):
        """Prepare sparse matrix for training"""
        # Convert sparse to dense
        X_dense = X_sparse.toarray() if hasattr(X_sparse, 'toarray') else X_sparse
        
        # Scale to [0, 1]
        X_scaled = self.scaler.fit_transform(X_dense)
        
        return X_scaled
    
    def train(self, X_train, X_val=None, epochs=50, batch_size=32):
        """Train the autoencoder"""
        if self.autoencoder is None:
            self.build_model()
        
        # Prepare data
        X_train_scaled = self.prepare_data(X_train)
        X_val_scaled = self.prepare_data(X_val) if X_val is not None else None
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'models/autoencoder/best_model.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            )
        ]
        
        # Train
        print(f"🚀 Training autoencoder for {epochs} epochs...")
        self.history = self.autoencoder.fit(
            X_train_scaled, X_train_scaled,
            validation_data=(X_val_scaled, X_val_scaled) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed")
        return self.history
    
    def encode(self, X):
        """Encode data to latent space"""
        X_scaled = self.prepare_data(X)
        return self.encoder.predict(X_scaled)
    
    def decode(self, latent):
        """Decode from latent space"""
        return self.decoder.predict(latent)
    
    def reconstruct(self, X):
        """Reconstruct input data"""
        X_scaled = self.prepare_data(X)
        return self.autoencoder.predict(X_scaled)
    
    def calculate_reconstruction_error(self, X):
        """Calculate reconstruction error for each sample"""
        X_scaled = self.prepare_data(X)
        X_reconstructed = self.autoencoder.predict(X_scaled)
        
        # MSE for each sample
        mse = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
        return mse
    
    def detect_anomalies(self, X, threshold_percentile=95):
        """Detect anomalies based on reconstruction error"""
        errors = self.calculate_reconstruction_error(X)
        threshold = np.percentile(errors, threshold_percentile)
        
        anomalies = errors > threshold
        
        print(f"🔍 Anomaly Detection Results:")
        print(f"   Threshold (95th percentile): {threshold:.4f}")
        print(f"   Anomalies found: {np.sum(anomalies)} ({np.mean(anomalies)*100:.2f}%)")
        
        return anomalies, errors, threshold
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Model MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('models/autoencoder/training_history.png')
        plt.show()
    
    def save_model(self, path='models/autoencoder/'):
        """Save the trained model"""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.autoencoder.save(os.path.join(path, 'autoencoder.h5'))
        self.encoder.save(os.path.join(path, 'encoder.h5'))
        self.decoder.save(os.path.join(path, 'decoder.h5'))
        print(f"✅ Models saved to {path}")
    
    def load_model(self, path='models/autoencoder/'):
        """Load a saved model"""
        import os
        self.autoencoder = keras.models.load_model(os.path.join(path, 'autoencoder.h5'))
        self.encoder = keras.models.load_model(os.path.join(path, 'encoder.h5'))
        self.decoder = keras.models.load_model(os.path.join(path, 'decoder.h5'))
        print(f"✅ Models loaded from {path}")