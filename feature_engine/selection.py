"""
Feature selection module for selecting top-k features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .scoring import FeatureScorer


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features based on scoring method.
    
    Parameters
    ----------
    k : int or float, default=10
        Number of top features to select. If float, interpreted as proportion.
    scoring_method : {'mutual_info', 'correlation'}, default='mutual_info'
        Method to use for scoring features.
    task : {'classification', 'regression'}, default='classification'
        Type of task (used for mutual information scoring).
    """
    
    def __init__(self, k=10, scoring_method='mutual_info', task='classification'):
        self.k = k
        self.scoring_method = scoring_method
        self.task = task
        self.selected_features_ = None
        self.scorer_ = None
        
    def fit(self, X, y):
        """
        Fit the feature selector.
        
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
            raise ValueError("FeatureSelector requires y for fitting")
            
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Initialize and fit scorer
        self.scorer_ = FeatureScorer(method=self.scoring_method, task=self.task)
        self.scorer_.fit(X_df, y)
        
        # Determine number of features to select
        if isinstance(self.k, float) and 0 < self.k < 1:
            # Interpret as proportion
            n_features = int(self.k * X_df.shape[1])
        else:
            n_features = int(self.k)
        
        # Ensure we don't select more features than available
        n_features = min(n_features, X_df.shape[1])
        
        # Get top features
        self.selected_features_ = self.scorer_.get_top_features(k=n_features)
        
        return self
    
    def transform(self, X):
        """
        Transform X by selecting top features.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_selected : DataFrame
            Data with only selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")
            
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Select only the top features
        return X_df[self.selected_features_]
    
    def get_feature_scores(self):
        """
        Get scores for all features.
        
        Returns
        -------
        scores : DataFrame
            Feature scores sorted by importance.
        """
        if self.scorer_ is None:
            raise ValueError("Selector has not been fitted yet")
            
        return self.scorer_.get_scores()
