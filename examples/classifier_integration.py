"""
Integration example with scikit-learn classifier.

This script demonstrates how to integrate the feature engineering pipeline
with a scikit-learn classifier.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from feature_engine import AutoFeatureEngineeringPipeline


def main():
    print("=" * 60)
    print("Classifier Integration Example")
    print("=" * 60)
    
    # Load a real dataset
    print("\n1. Loading breast cancer dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a pipeline with feature engineering and classifier
    print("\n2. Creating pipeline with feature engineering...")
    
    # Pipeline with feature engineering
    pipeline_with_fe = Pipeline([
        ('feature_engineering', AutoFeatureEngineeringPipeline(
            use_polynomial=True,
            use_binning=True,
            use_target_encoding=False,  # No categorical features
            polynomial_degree=2,
            n_bins=5,
            k_features=30,
            scoring_method='mutual_info',
            task='classification'
        )),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Pipeline without feature engineering (baseline)
    pipeline_baseline = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train and evaluate both pipelines
    print("\n3. Training and evaluating models...")
    
    # Baseline model
    print("\n   Baseline (no feature engineering):")
    pipeline_baseline.fit(X_train, y_train)
    baseline_train_score = pipeline_baseline.score(X_train, y_train)
    baseline_test_score = pipeline_baseline.score(X_test, y_test)
    
    print(f"   Training accuracy: {baseline_train_score:.4f}")
    print(f"   Test accuracy: {baseline_test_score:.4f}")
    
    # Model with feature engineering
    print("\n   With feature engineering:")
    pipeline_with_fe.fit(X_train, y_train)
    fe_train_score = pipeline_with_fe.score(X_train, y_train)
    fe_test_score = pipeline_with_fe.score(X_test, y_test)
    
    print(f"   Training accuracy: {fe_train_score:.4f}")
    print(f"   Test accuracy: {fe_test_score:.4f}")
    
    # Cross-validation comparison
    print("\n4. Cross-validation comparison (5-fold)...")
    
    cv_baseline = cross_val_score(
        pipeline_baseline, X_train, y_train, cv=5, scoring='accuracy'
    )
    cv_with_fe = cross_val_score(
        pipeline_with_fe, X_train, y_train, cv=5, scoring='accuracy'
    )
    
    print(f"\n   Baseline CV accuracy: {cv_baseline.mean():.4f} (+/- {cv_baseline.std():.4f})")
    print(f"   With FE CV accuracy: {cv_with_fe.mean():.4f} (+/- {cv_with_fe.std():.4f})")
    
    # Show selected features
    print("\n5. Top selected features:")
    fe_step = pipeline_with_fe.named_steps['feature_engineering']
    try:
        selected_features = fe_step.get_feature_names()
        if selected_features:
            print(f"   Number of selected features: {len(selected_features)}")
            print(f"   First 10: {selected_features[:10]}")
    except:
        print("   (Unable to retrieve feature names)")
    
    # Example with Logistic Regression
    print("\n6. Trying with Logistic Regression...")
    
    lr_with_fe = Pipeline([
        ('feature_engineering', AutoFeatureEngineeringPipeline(
            use_polynomial=True,
            use_binning=False,
            use_target_encoding=False,
            polynomial_degree=2,
            k_features=25,
            scoring_method='correlation',
            task='classification'
        )),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    lr_with_fe.fit(X_train, y_train)
    lr_test_score = lr_with_fe.score(X_test, y_test)
    
    print(f"   Logistic Regression test accuracy: {lr_test_score:.4f}")
    
    print("\n" + "=" * 60)
    print("Integration example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
