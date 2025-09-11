import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.autoencoder.autoencoder import CabinetDecisionAutoencoder
from sklearn.model_selection import train_test_split
import pickle

# Load preprocessed data
with open('data/tfidf_matrix.pkl', 'rb') as f:
    X = pickle.load(f)

print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")

# Split the data
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train autoencoder
model = CabinetDecisionAutoencoder(input_dim=X.shape[1], encoding_dim=64)
model.build_model()
model.train(X_train)  # Train only on training set

# Evaluate on test set
print("\nEvaluating on test set...")
anomalies, errors, threshold = model.detect_anomalies(X_test)

# Save model
model.save_model()