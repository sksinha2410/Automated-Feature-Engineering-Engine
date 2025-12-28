"""
Tests for the automated feature engineering pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_engine.pipeline import AutoFeatureEngineeringPipeline


class TestAutoFeatureEngineeringPipeline:
    """Tests for AutoFeatureEngineeringPipeline."""
    
    def test_basic_pipeline(self):
        """Test basic pipeline functionality."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        
        pipeline = AutoFeatureEngineeringPipeline(
            use_polynomial=True,
            use_binning=True,
            use_target_encoding=False,
            k_features=10
        )
        
        pipeline.fit(X_df, y)
        X_transformed = pipeline.transform(X_df)
        
        # Should select top k features
        assert X_transformed.shape[1] == 10
        assert X_transformed.shape[0] == X_df.shape[0]
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X)
        
        pipeline = AutoFeatureEngineeringPipeline(k_features=15)
        X_transformed = pipeline.fit_transform(X_df, y)
        
        assert X_transformed.shape[0] == X_df.shape[0]
        assert X_transformed.shape[1] <= 15
    
    def test_without_selection(self):
        """Test pipeline without feature selection."""
        X = pd.DataFrame(np.random.randn(50, 5))
        
        pipeline = AutoFeatureEngineeringPipeline(
            use_polynomial=True,
            use_binning=False,
            use_target_encoding=False
        )
        
        # Fit without y (no selection) - create a dummy y just for fitting
        # but the pipeline should still generate features
        y_dummy = np.random.randint(0, 2, 50)
        pipeline.fit(X, y_dummy)
        X_transformed = pipeline.transform(X)
        
        # Should have selected features (since we provided y)
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_with_categorical_features(self):
        """Test pipeline with categorical features."""
        X = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
            'cat2': np.random.choice(['X', 'Y'], 100)
        })
        y = np.random.randint(0, 2, 100)
        
        pipeline = AutoFeatureEngineeringPipeline(
            use_polynomial=True,
            use_binning=True,
            use_target_encoding=True,
            k_features=20
        )
        
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] <= 20
