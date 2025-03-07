#!/usr/bin/env python
# coding: utf-8

# # Model Ensemble Module Documentation
# 
# This module (`model_ensemble.py`) provides a function to train a Random Forest classifier with hyperparameter tuning using Grid Search. By exploring a grid of parameters, this approach aims to optimize model performance (measured via ROC AUC) and select the best model configuration.
# 
# ## Overview
# 
# The function in this module performs the following steps:
# 
# - **Instantiate the Classifier:**  
#   Creates a `RandomForestClassifier` with a fixed `random_state` for reproducibility.
# 
# - **Define Hyperparameter Grid:**  
#   Specifies a grid of parameters to tune:
#   - `n_estimators`: Number of trees in the forest.
#   - `max_depth`: Maximum depth of each tree (with `None` indicating no limit).
#   - `min_samples_split`: Minimum number of samples required to split an internal node.
# 
# - **Grid Search with Cross-Validation:**  
#   Uses `GridSearchCV` to search for the best combination of hyperparameters over 3-fold cross-validation, optimizing for the ROC AUC metric.
# 
# - **Fit and Return Best Estimator:**  
#   Fits the grid search on the training data, prints the best parameters, and returns the best estimator for further use.
# 
# ## Function Details
# 
# ### `train_random_forest(X_train, y_train)`
# 
# - **Purpose:**  
#   Train a Random Forest model with hyperparameter tuning to achieve better predictive performance.
# 
# - **Parameters:**  
#   - `X_train`: Feature matrix for the training set.
#   - `y_train`: Target labels for the training set.
# 
# - **Returns:**  
#   The best estimator found by `GridSearchCV`, which can be used to make predictions on new data.
# 
# - **Usage Example:**
#   ```python
#   from src.model_ensemble import train_random_forest
# 
#   # X_train and y_train should be your training features and labels.
#   best_rf_model = train_random_forest(X_train, y_train)
# 

# In[1]:


# src/model_ensemble.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model using Grid Search for hyperparameter tuning.
    """
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_rf = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    
    print("Best parameters found:", grid_rf.best_params_)
    return grid_rf.best_estimator_

