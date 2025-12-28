"""
Feature transformation classes for automated feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer


class PolynomialTransformer(BaseEstimator, TransformerMixin):
    """
    Generate polynomial features from numeric columns.
    
    Parameters
    ----------
    degree : int, default=2
        The degree of polynomial features to generate.
    interaction_only : bool, default=False
        If True, only interaction features are produced.
    include_bias : bool, default=False
        If True, include a bias column.
    """
    
    def __init__(self, degree=2, interaction_only=False, include_bias=False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly_ = None
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        """
        Fit the polynomial transformer.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like, optional
            Target values (not used).
            
        Returns
        -------
        self : object
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Only use numeric columns
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        self.numeric_cols_ = numeric_cols
        
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        self.poly_.fit(X_df[numeric_cols])
        
        # Generate feature names
        if hasattr(X_df, 'columns'):
            input_features = [str(col) for col in numeric_cols]
        else:
            input_features = [f"x{i}" for i in range(len(numeric_cols))]
            
        self.feature_names_ = self.poly_.get_feature_names_out(input_features)
        
        return self
    
    def transform(self, X):
        """
        Transform X using polynomial features.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : DataFrame
            Transformed data with polynomial features.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        X_poly = self.poly_.transform(X_df[self.numeric_cols_])
        
        # Create DataFrame with feature names
        X_poly_df = pd.DataFrame(
            X_poly,
            columns=self.feature_names_,
            index=X_df.index
        )
        
        return X_poly_df


class BinningTransformer(BaseEstimator, TransformerMixin):
    """
    Bin continuous features into discrete intervals.
    
    Parameters
    ----------
    n_bins : int, default=5
        Number of bins to produce.
    encode : {'onehot', 'ordinal'}, default='onehot'
        Method used to encode the transformed result.
    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.
    """
    
    def __init__(self, n_bins=5, encode='onehot', strategy='quantile'):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.discretizer_ = None
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        """
        Fit the binning transformer.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like, optional
            Target values (not used).
            
        Returns
        -------
        self : object
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Only use numeric columns
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        self.numeric_cols_ = numeric_cols
        
        self.discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode=self.encode,
            strategy=self.strategy
        )
        
        self.discretizer_.fit(X_df[numeric_cols])
        
        # Generate feature names
        if self.encode == 'onehot':
            feature_names = []
            for i, col in enumerate(numeric_cols):
                for bin_idx in range(self.n_bins):
                    feature_names.append(f"{col}_bin_{bin_idx}")
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f"{col}_binned" for col in numeric_cols]
        
        return self
    
    def transform(self, X):
        """
        Transform X using binning.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : DataFrame
            Transformed data with binned features.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        X_binned = self.discretizer_.transform(X_df[self.numeric_cols_])
        
        # Convert to dense array if sparse
        if hasattr(X_binned, 'toarray'):
            X_binned = X_binned.toarray()
        
        # Create DataFrame with feature names
        X_binned_df = pd.DataFrame(
            X_binned,
            columns=self.feature_names_,
            index=X_df.index
        )
        
        return X_binned_df


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features using target statistics.
    
    Parameters
    ----------
    smoothing : float, default=1.0
        Smoothing parameter for regularization.
    min_samples_leaf : int, default=1
        Minimum samples to take category average into account.
    """
    
    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encodings_ = {}
        self.global_mean_ = None
        
    def fit(self, X, y):
        """
        Fit the target encoder.
        
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
            raise ValueError("Target encoder requires y for fitting")
            
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_series = pd.Series(y)
        
        # Store global mean
        self.global_mean_ = y_series.mean()
        
        # Identify categorical columns
        categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns
        self.categorical_cols_ = categorical_cols
        
        # Calculate encodings for each categorical column
        for col in categorical_cols:
            # Calculate mean target for each category
            agg = pd.DataFrame({'category': X_df[col], 'target': y_series})
            stats = agg.groupby('category')['target'].agg(['mean', 'count'])
            
            # Apply smoothing
            smoothing_factor = 1 / (1 + np.exp(-(stats['count'] - self.min_samples_leaf) / self.smoothing))
            stats['smoothed_mean'] = (
                self.global_mean_ * (1 - smoothing_factor) + 
                stats['mean'] * smoothing_factor
            )
            
            self.encodings_[col] = stats['smoothed_mean'].to_dict()
        
        return self
    
    def transform(self, X):
        """
        Transform X using target encoding.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : DataFrame
            Transformed data with target-encoded features.
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        X_encoded = X_df.copy()
        
        for col in self.categorical_cols_:
            # Map categories to their encodings, use global mean for unknown categories
            X_encoded[f"{col}_target_enc"] = X_df[col].map(
                self.encodings_[col]
            ).fillna(self.global_mean_)
        
        # Return only the encoded columns
        encoded_cols = [f"{col}_target_enc" for col in self.categorical_cols_]
        
        return X_encoded[encoded_cols]
