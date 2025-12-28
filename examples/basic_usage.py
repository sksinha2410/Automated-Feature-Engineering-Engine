"""
Basic usage example of the Automated Feature Engineering Engine.

This script demonstrates how to use the feature engineering library
with a simple dataset.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from feature_engine import (
    PolynomialTransformer,
    BinningTransformer,
    FeatureScorer,
    FeatureSelector
)


def main():
    print("=" * 60)
    print("Basic Feature Engineering Example")
    print("=" * 60)
    
    # Generate a synthetic dataset
    print("\n1. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"   Dataset shape: {X_df.shape}")
    print(f"   Features: {list(X_df.columns)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Example 1: Polynomial Features
    print("\n2. Generating polynomial features...")
    poly_transformer = PolynomialTransformer(degree=2, include_bias=False)
    poly_transformer.fit(X_train)
    X_train_poly = poly_transformer.transform(X_train)
    
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   After polynomial transformation: {X_train_poly.shape[1]}")
    print(f"   Sample feature names: {list(X_train_poly.columns[:5])}")
    
    # Example 2: Binning Features
    print("\n3. Creating binned features...")
    binning_transformer = BinningTransformer(n_bins=5, encode='onehot')
    binning_transformer.fit(X_train)
    X_train_binned = binning_transformer.transform(X_train)
    
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   After binning: {X_train_binned.shape[1]}")
    print(f"   Sample feature names: {list(X_train_binned.columns[:5])}")
    
    # Example 3: Feature Scoring
    print("\n4. Scoring features with mutual information...")
    scorer = FeatureScorer(method='mutual_info', task='classification')
    scorer.fit(X_train_poly, y_train)
    
    scores_df = scorer.get_scores()
    print(f"\n   Top 10 features by mutual information:")
    print(scores_df.head(10).to_string(index=False))
    
    # Example 4: Feature Selection
    print("\n5. Selecting top features...")
    selector = FeatureSelector(k=15, scoring_method='mutual_info', task='classification')
    selector.fit(X_train_poly, y_train)
    X_train_selected = selector.transform(X_train_poly)
    
    print(f"   Features before selection: {X_train_poly.shape[1]}")
    print(f"   Features after selection: {X_train_selected.shape[1]}")
    print(f"   Selected features: {selector.selected_features_}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
