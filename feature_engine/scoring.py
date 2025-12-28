"""
Feature scoring module for ranking features based on their importance.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.base import BaseEstimator


class FeatureScorer(BaseEstimator):
    """
    Score features using mutual information or correlation.
    
    Parameters
    ----------
    method : {'mutual_info', 'correlation'}, default='mutual_info'
        Method to use for scoring features.
    task : {'classification', 'regression'}, default='classification'
        Type of task (used for mutual information scoring).
    """
    
    def __init__(self, method='mutual_info', task='classification'):
        self.method = method
        self.task = task
        self.scores_ = None
        self.feature_names_ = None
        
    def fit(self, X, y):
        """
        Compute feature scores.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
        """
        if y is None:
            raise ValueError("FeatureScorer requires y for fitting")
            
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_array = np.array(y).ravel()
        
        # Convert object dtypes to numeric where possible
        for col in X_df.columns:
            if X_df[col].dtype == 'object':
                try:
                    X_df[col] = pd.to_numeric(X_df[col])
                except (ValueError, TypeError):
                    # Keep as object if conversion fails
                    pass
        
        # Only use numeric columns for scoring
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric features found for scoring. All features are non-numeric.")
        
        X_numeric = X_df[numeric_cols]
        
        # Store feature names
        if hasattr(X_numeric, 'columns'):
            self.feature_names_ = list(X_numeric.columns)
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X_numeric.shape[1])]
        
        if self.method == 'mutual_info':
            # Use mutual information
            if self.task == 'classification':
                scores = mutual_info_classif(X_numeric, y_array, random_state=42)
            else:
                scores = mutual_info_regression(X_numeric, y_array, random_state=42)
        elif self.method == 'correlation':
            # Use correlation with target
            scores = []
            for col in X_numeric.columns:
                try:
                    corr = np.abs(np.corrcoef(X_numeric[col], y_array)[0, 1])
                    # Handle NaN (constant features)
                    scores.append(corr if not np.isnan(corr) else 0.0)
                except:
                    scores.append(0.0)
            scores = np.array(scores)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.scores_ = scores
        
        return self
    
    def get_scores(self, as_dataframe=True):
        """
        Get feature scores.
        
        Parameters
        ----------
        as_dataframe : bool, default=True
            If True, return scores as a DataFrame, otherwise as array.
            
        Returns
        -------
        scores : DataFrame or array
            Feature scores.
        """
        if self.scores_ is None:
            raise ValueError("Scorer has not been fitted yet")
            
        if as_dataframe:
            return pd.DataFrame({
                'feature': self.feature_names_,
                'score': self.scores_
            }).sort_values('score', ascending=False)
        else:
            return self.scores_
    
    def get_top_features(self, k=10):
        """
        Get top k features by score.
        
        Parameters
        ----------
        k : int, default=10
            Number of top features to return.
            
        Returns
        -------
        top_features : list
            Names of top k features.
        """
        if self.scores_ is None:
            raise ValueError("Scorer has not been fitted yet")
            
        # Get indices of top k scores
        top_indices = np.argsort(self.scores_)[::-1][:k]
        
        return [self.feature_names_[i] for i in top_indices]
