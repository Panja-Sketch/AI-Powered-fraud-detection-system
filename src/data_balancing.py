#!/usr/bin/env python
# coding: utf-8

# # Data Balancing Module Documentation
# 
# This module (`data_balancing.py`) provides a utility to address imbalanced datasets using SMOTE (Synthetic Minority Over-sampling Technique). Balancing the data is crucial for tasks like fraud detection, where one class (e.g., fraudulent transactions) is significantly underrepresented compared to the other.
# 
# ## Overview
# 
# Imbalanced data can lead to biased models that perform poorly on the minority class. The `balance_data` function leverages SMOTE to create synthetic examples of the minority class, resulting in a more balanced dataset. This process can improve the performance of machine learning models by ensuring that the classifier has enough examples from both classes during training.
# 
# ## Function Details
# 
# ### `balance_data(X, y)`
# 
# - **Purpose:**  
#   Balances the dataset by oversampling the minority class using SMOTE.
# 
# - **Parameters:**  
#   - `X`: The feature set (e.g., a Pandas DataFrame or NumPy array) containing the predictors.
#   - `y`: The target variable (e.g., a Pandas Series or NumPy array) containing class labels.
# 
# - **Returns:**  
#   A tuple `(X_res, y_res)` where:
#   - `X_res` is the resampled feature set.
#   - `y_res` is the resampled target variable.
# 
# - **Usage Example:**
#   ```python
#   from src.data_balancing import balance_data
# 
#   # Assume X and y are your feature set and target variable, respectively.
#   X_res, y_res = balance_data(X, y)
# 

# In[1]:


# src/data_balancing.py
from imblearn.over_sampling import SMOTE

def balance_data(X, y):
    """
    Balance the dataset using SMOTE to oversample the minority class.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

