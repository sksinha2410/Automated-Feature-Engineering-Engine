"""
Tests for feature transformers.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_engine.transformers import (
    PolynomialTransformer,
    BinningTransformer,
    TargetEncoder
)


class TestPolynomialTransformer:
    """Tests for PolynomialTransformer."""
    
    def test_basic_transformation(self):
        """Test basic polynomial transformation."""
        X = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [2, 3, 4, 5]
        })
        
        transformer = PolynomialTransformer(degree=2, include_bias=False)
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        
        # Should have more features than input
        assert X_transformed.shape[1] > X.shape[1]
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_with_array_input(self):
        """Test with numpy array input."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        transformer = PolynomialTransformer(degree=2)
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        
        assert X_transformed.shape[0] == X.shape[0]
        assert isinstance(X_transformed, pd.DataFrame)


class TestBinningTransformer:
    """Tests for BinningTransformer."""
    
    def test_onehot_binning(self):
        """Test binning with one-hot encoding."""
        X = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100)
        })
        
        transformer = BinningTransformer(n_bins=5, encode='onehot')
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        
        # With 2 features and 5 bins each, should have 10 columns
        assert X_transformed.shape[1] == 10
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_ordinal_binning(self):
        """Test binning with ordinal encoding."""
        X = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100)
        })
        
        transformer = BinningTransformer(n_bins=5, encode='ordinal')
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        
        # Ordinal encoding preserves number of features
        assert X_transformed.shape[1] == X.shape[1]
        assert X_transformed.shape[0] == X.shape[0]


class TestTargetEncoder:
    """Tests for TargetEncoder."""
    
    def test_basic_encoding(self):
        """Test basic target encoding."""
        X = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'C', 'C', 'B']
        })
        y = np.array([1, 0, 1, 0, 1, 1, 0, 1])
        
        encoder = TargetEncoder()
        encoder.fit(X, y)
        X_transformed = encoder.transform(X)
        
        assert X_transformed.shape[0] == X.shape[0]
        assert 'category_target_enc' in X_transformed.columns
    
    def test_requires_target(self):
        """Test that fit requires y."""
        X = pd.DataFrame({'category': ['A', 'B', 'A']})
        
        encoder = TargetEncoder()
        with pytest.raises(ValueError):
            encoder.fit(X, None)
    
    def test_unseen_categories(self):
        """Test handling of unseen categories."""
        X_train = pd.DataFrame({'category': ['A', 'B', 'A', 'B']})
        y_train = np.array([1, 0, 1, 0])
        
        X_test = pd.DataFrame({'category': ['A', 'C']})  # C is unseen
        
        encoder = TargetEncoder()
        encoder.fit(X_train, y_train)
        X_test_transformed = encoder.transform(X_test)
        
        # Unseen category should get global mean
        assert X_test_transformed.shape[0] == X_test.shape[0]
        assert not X_test_transformed.isnull().any().any()
