"""
Tests for feature scoring and selection.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_engine.scoring import FeatureScorer
from feature_engine.selection import FeatureSelector


class TestFeatureScorer:
    """Tests for FeatureScorer."""
    
    def test_mutual_info_scoring(self):
        """Test mutual information scoring."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X)
        
        scorer = FeatureScorer(method='mutual_info', task='classification')
        scorer.fit(X_df, y)
        
        scores = scorer.get_scores(as_dataframe=False)
        assert len(scores) == X_df.shape[1]
        assert all(scores >= 0)  # MI scores are non-negative
    
    def test_correlation_scoring(self):
        """Test correlation scoring."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X)
        
        scorer = FeatureScorer(method='correlation', task='classification')
        scorer.fit(X_df, y)
        
        scores = scorer.get_scores(as_dataframe=False)
        assert len(scores) == X_df.shape[1]
        assert all(scores >= 0)  # Using absolute correlation
    
    def test_get_top_features(self):
        """Test getting top features."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
        
        scorer = FeatureScorer(method='mutual_info')
        scorer.fit(X_df, y)
        
        top_features = scorer.get_top_features(k=5)
        assert len(top_features) == 5
        assert all(isinstance(f, str) for f in top_features)
    
    def test_requires_target(self):
        """Test that fit requires y."""
        X = pd.DataFrame(np.random.randn(10, 5))
        
        scorer = FeatureScorer()
        with pytest.raises(ValueError):
            scorer.fit(X, None)


class TestFeatureSelector:
    """Tests for FeatureSelector."""
    
    def test_basic_selection(self):
        """Test basic feature selection."""
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        X_df = pd.DataFrame(X)
        
        selector = FeatureSelector(k=10)
        selector.fit(X_df, y)
        X_selected = selector.transform(X_df)
        
        assert X_selected.shape[1] == 10
        assert X_selected.shape[0] == X_df.shape[0]
    
    def test_proportion_selection(self):
        """Test selection with proportion."""
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        X_df = pd.DataFrame(X)
        
        selector = FeatureSelector(k=0.5)  # Select 50%
        selector.fit(X_df, y)
        X_selected = selector.transform(X_df)
        
        assert X_selected.shape[1] == 10  # 50% of 20
        assert X_selected.shape[0] == X_df.shape[0]
    
    def test_get_feature_scores(self):
        """Test getting feature scores."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_df = pd.DataFrame(X)
        
        selector = FeatureSelector(k=5)
        selector.fit(X_df, y)
        
        scores_df = selector.get_feature_scores()
        assert isinstance(scores_df, pd.DataFrame)
        assert 'feature' in scores_df.columns
        assert 'score' in scores_df.columns
