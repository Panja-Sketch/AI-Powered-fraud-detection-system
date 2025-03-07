#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection Module Documentation
# 
# This module (`anomaly_detection.py`) provides two key functions for unsupervised anomaly detection:
# 1. **Isolation Forest:** An algorithm to detect anomalies by isolating observations.
# 2. **Autoencoder:** A neural network model built with Keras for learning compressed representations and identifying anomalies via reconstruction error.
# 
# ## Overview
# 
# Anomaly detection is crucial in scenarios such as fraud detection, where unusual patterns or outliers may indicate fraudulent activity. This module includes:
# 
# - **`train_isolation_forest(X)`**:  
#   Trains an Isolation Forest model on the feature data to flag anomalies.
# 
# - **`build_autoencoder(input_dim)`**:  
#   Constructs a simple autoencoder using TensorFlow and Keras. The autoencoder learns to reconstruct normal input data, and high reconstruction error on unseen data can be used as an indicator of anomalies.
# 
# ## Function Details
# 
# ### `train_isolation_forest(X)`
# 
# - **Purpose:**  
#   Train an Isolation Forest model to detect anomalies within the dataset.
# 
# - **Parameters:**  
#   - `X`: Feature matrix (e.g., a NumPy array or Pandas DataFrame) used for training the Isolation Forest.
# 
# - **Returns:**  
#   - A trained `IsolationForest` model that can be used to predict anomaly scores or identify outliers.
# 
# - **Usage Example:**
#   ```python
#   from src.anomaly_detection import train_isolation_forest
#   
#   # Assume X is your feature matrix
#   iso_forest = train_isolation_forest(X)
#   anomaly_scores = iso_forest.decision_function(X_test)
#   print("Anomaly scores:", anomaly_scores)
# 

# In[1]:


# src/anomaly_detection.py
import numpy as np
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def train_isolation_forest(X):
    """
    Train an Isolation Forest to detect anomalies.
    """
    iso_forest = IsolationForest(contamination=0.001, random_state=42)
    iso_forest.fit(X)
    return iso_forest

def build_autoencoder(input_dim):
    """
    Build a simple autoencoder using Keras.
    """
    # Define the input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder: compress input to a smaller representation
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    bottleneck = Dense(8, activation='relu')(encoded)
    
    # Decoder: reconstruct the original input
    decoded = Dense(16, activation='relu')(bottleneck)
    decoded = Dense(32, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    # Construct and compile the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

