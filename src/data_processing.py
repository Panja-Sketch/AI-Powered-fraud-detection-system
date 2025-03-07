#!/usr/bin/env python
# coding: utf-8

# # Data Processing Module Documentation
# 
# This module (`data_processing.py`) contains essential functions for loading and preprocessing your dataset. It is designed to prepare raw data for subsequent analysis or modeling, particularly in contexts like fraud detection.
# 
# ## Overview
# 
# The module provides two main functions:
# - **`load_data(file_path)`**: Loads a CSV file into a Pandas DataFrame.
# - **`feature_engineering(df)`**: Enhances the dataset by creating new features that may improve model performance.
# 
# ## Function Details
# 
# ### `load_data(file_path)`
# 
# - **Purpose:**  
#   Loads the dataset from a CSV file into a Pandas DataFrame for further processing.
# 
# - **Parameters:**  
#   - `file_path` (str): The path to the CSV file containing the dataset.
# 
# - **Returns:**  
#   A Pandas DataFrame containing the loaded data.
# 
# - **Usage Example:**
#   ```python
#   df = load_data('data/creditcard.csv')
# 
# Summary-
# The data_processing.py module is a critical component in your data pipeline. By loading and transforming the raw data, it helps ensure that subsequent analyses and machine learning models are built on a well-prepared foundation. This preprocessing step can lead to better model performance, particularly when dealing with complex datasets such as those found in fraud detection tasks.
# 

# In[1]:


# src/data_processing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def feature_engineering(df):
    """
    Create new features:
    - HourOfDay: Derived from 'Time'
    - Amount_scaled: Standardized 'Amount'
    - RollingMeanAmount: A rolling mean on 'Amount'
    """
    # Create a temporal feature: HourOfDay
    df['HourOfDay'] = (df['Time'] / 3600) % 24

    # Standardize the 'Amount' feature
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    
    # Rolling mean of Amount over a window of 5 transactions
    df['RollingMeanAmount'] = df['Amount'].rolling(window=5, min_periods=1).mean()
    
    return df


# In[ ]:




