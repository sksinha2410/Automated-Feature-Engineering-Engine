"""
Pipeline integration with scikit-learn for automated feature engineering.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from .transformers import PolynomialTransformer, BinningTransformer, TargetEncoder
from .selection import FeatureSelector


class AutoFeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    Automated feature engineering pipeline that combines multiple transformations
    and selects the best features.
    
    Parameters
    ----------
    use_polynomial : bool, default=True
        Whether to generate polynomial features.
    use_binning : bool, default=True
        Whether to generate binned features.
    use_target_encoding : bool, default=True
        Whether to use target encoding for categorical features.
    polynomial_degree : int, default=2
        Degree of polynomial features.
    n_bins : int, default=5
        Number of bins for binning transformer.
    k_features : int or float, default=20
        Number of top features to select.
    scoring_method : {'mutual_info', 'correlation'}, default='mutual_info'
        Method to use for feature scoring.
    task : {'classification', 'regression'}, default='classification'
        Type of machine learning task.
    """
    
    def __init__(
        self,
        use_polynomial=True,
        use_binning=True,
        use_target_encoding=True,
        polynomial_degree=2,
        n_bins=5,
        k_features=20,
        scoring_method='mutual_info',
        task='classification'
    ):
        self.use_polynomial = use_polynomial
        self.use_binning = use_binning
        self.use_target_encoding = use_target_encoding
        self.polynomial_degree = polynomial_degree
        self.n_bins = n_bins
        self.k_features = k_features
        self.scoring_method = scoring_method
        self.task = task
        self.pipeline_ = None
        
    def fit(self, X, y=None):
        """
        Fit the auto feature engineering pipeline.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
            
        Returns
        -------
        self : object
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Build list of transformers to combine
        transformers = []
        
        # Add original features (pass-through)
        transformers.append(('original', PassThroughTransformer()))
        
        # Add polynomial features if requested
        if self.use_polynomial:
            transformers.append((
                'polynomial',
                PolynomialTransformer(degree=self.polynomial_degree)
            ))
        
        # Add binning features if requested
        if self.use_binning:
            transformers.append((
                'binning',
                BinningTransformer(n_bins=self.n_bins)
            ))
        
        # Add target encoding if requested
        if self.use_target_encoding:
            # Check if there are categorical columns
            cat_cols = X_df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0 and y is not None:
                transformers.append((
                    'target_encoding',
                    TargetEncoder()
                ))
        
        # Combine all transformers
        feature_union = FeatureUnion(transformers)
        
        # Create full pipeline with feature selection
        steps = [
            ('feature_generation', feature_union),
        ]
        
        # Add feature selection if y is provided
        if y is not None:
            steps.append((
                'feature_selection',
                FeatureSelector(
                    k=self.k_features,
                    scoring_method=self.scoring_method,
                    task=self.task
                )
            ))
        
        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(X_df, y)
        
        return self
    
    def transform(self, X):
        """
        Transform X using the fitted pipeline.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : DataFrame
            Transformed data with engineered features.
        """
        if self.pipeline_ is None:
            raise ValueError("Pipeline has not been fitted yet")
            
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        return self.pipeline_.transform(X_df)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
            
        Returns
        -------
        X_transformed : DataFrame
            Transformed training data.
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self):
        """
        Get names of selected features.
        
        Returns
        -------
        feature_names : list or None
            Names of selected features, or None if pipeline not fitted.
        """
        if self.pipeline_ is None:
            raise ValueError("Pipeline has not been fitted yet")
            
        # Try to get feature names from selector
        if 'feature_selection' in self.pipeline_.named_steps:
            return self.pipeline_.named_steps['feature_selection'].selected_features_
        else:
            # If no selection, we cannot easily determine feature names from FeatureUnion
            return None


class PassThroughTransformer(BaseEstimator, TransformerMixin):
    """
    Simple pass-through transformer that returns input unchanged.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X_df
