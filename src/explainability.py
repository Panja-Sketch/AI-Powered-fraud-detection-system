#!/usr/bin/env python
# coding: utf-8

# # Explainability Module Documentation
# 
# This module (`explainability.py`) provides a function to generate visual explanations for tree-based models using SHAP. It is particularly useful in understanding how individual features contribute to a model's predictions, which can be essential for model validation and transparency.
# 
# ## Overview
# 
# The module contains the function `explain_model(model, X_sample)`, which leverages SHAP's `TreeExplainer` to calculate SHAP values for a sample of the input data and then displays a summary plot of feature importance.
# 
# ## Function Details
# 
# ### `explain_model(model, X_sample)`
# 
# - **Purpose:**  
#   To explain the predictions of a tree-based model by calculating and visualizing SHAP values, which illustrate the contribution of each feature to the model's outputs.
# 
# - **Parameters:**  
#   - `model`: A trained tree-based machine learning model (e.g., `RandomForestClassifier`) used to generate predictions.
#   - `X_sample`: A subset of the input data (as a Pandas DataFrame or NumPy array) on which you want to generate SHAP explanations.
# 
# - **Returns:**  
#   The function does not return any value; it displays a SHAP summary plot that highlights feature importance.
# 
# - **Usage Example:**
#   ```python
#   from src.explainability import explain_model
#   
#   # Assume rf_model is a trained RandomForestClassifier and X_test is your test data
#   X_sample = X_test.sample(100, random_state=42)  # Select a random sample for explanation
#   explain_model(rf_model, X_sample)
# 

# In[1]:


# src/explainability.py
import shap

def explain_model(model, X_sample):
    """
    Use SHAP to explain predictions of a tree-based model.
    """
    # Create a TreeExplainer for the model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Display a summary plot of feature importance
    shap.summary_plot(shap_values, X_sample)

