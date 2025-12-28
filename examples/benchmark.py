"""
Benchmark script comparing raw features vs. engineered features.

This script provides a comprehensive benchmark showing the impact of
automated feature engineering on model performance.
"""

import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from feature_engine import AutoFeatureEngineeringPipeline


def benchmark_dataset(dataset_name, X, y, classifiers):
    """
    Benchmark a dataset with and without feature engineering.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    X : DataFrame
        Features
    y : array
        Target
    classifiers : dict
        Dictionary of classifier name -> classifier object
        
    Returns
    -------
    results : DataFrame
        Benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {dataset_name}")
    print(f"{'='*60}")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = []
    
    for clf_name, clf in classifiers.items():
        print(f"\n{clf_name}:")
        
        # Baseline (raw features)
        print("  Training baseline (raw features)...")
        start_time = time.time()
        clf.fit(X_train, y_train)
        baseline_train_time = time.time() - start_time
        
        y_pred_baseline = clf.predict(X_test)
        baseline_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_baseline),
            'f1': f1_score(y_test, y_pred_baseline, average='weighted'),
            'precision': precision_score(y_test, y_pred_baseline, average='weighted'),
            'recall': recall_score(y_test, y_pred_baseline, average='weighted')
        }
        
        print(f"    Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"    F1-score: {baseline_metrics['f1']:.4f}")
        print(f"    Training time: {baseline_train_time:.3f}s")
        
        # With feature engineering
        print("  Training with feature engineering...")
        
        # Create feature engineering pipeline
        fe_pipeline = AutoFeatureEngineeringPipeline(
            use_polynomial=True,
            use_binning=True,
            use_target_encoding=False,
            polynomial_degree=2,
            n_bins=5,
            k_features=min(30, X.shape[1] * 3),  # Adaptive k
            scoring_method='mutual_info',
            task='classification'
        )
        
        start_time = time.time()
        X_train_fe = fe_pipeline.fit_transform(X_train, y_train)
        X_test_fe = fe_pipeline.transform(X_test)
        
        # Train classifier on engineered features
        clf_fe = clf.__class__(**clf.get_params())
        clf_fe.fit(X_train_fe, y_train)
        fe_train_time = time.time() - start_time
        
        y_pred_fe = clf_fe.predict(X_test_fe)
        fe_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_fe),
            'f1': f1_score(y_test, y_pred_fe, average='weighted'),
            'precision': precision_score(y_test, y_pred_fe, average='weighted'),
            'recall': recall_score(y_test, y_pred_fe, average='weighted')
        }
        
        print(f"    Accuracy: {fe_metrics['accuracy']:.4f}")
        print(f"    F1-score: {fe_metrics['f1']:.4f}")
        print(f"    Training time: {fe_train_time:.3f}s")
        print(f"    Features: {X_train.shape[1]} -> {X_train_fe.shape[1]}")
        
        # Calculate improvements
        accuracy_improvement = (fe_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
        f1_improvement = (fe_metrics['f1'] - baseline_metrics['f1']) * 100
        
        print(f"    Improvement: Accuracy +{accuracy_improvement:.2f}%, F1 +{f1_improvement:.2f}%")
        
        # Store results
        results.append({
            'Dataset': dataset_name,
            'Classifier': clf_name,
            'Baseline_Accuracy': baseline_metrics['accuracy'],
            'FE_Accuracy': fe_metrics['accuracy'],
            'Baseline_F1': baseline_metrics['f1'],
            'FE_F1': fe_metrics['f1'],
            'Accuracy_Improvement': accuracy_improvement,
            'F1_Improvement': f1_improvement,
            'Original_Features': X_train.shape[1],
            'Engineered_Features': X_train_fe.shape[1],
            'Baseline_Time': baseline_train_time,
            'FE_Time': fe_train_time
        })
    
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Feature Engineering Benchmark")
    print("=" * 60)
    print("\nComparing model performance with raw vs. engineered features")
    
    # Define classifiers to benchmark
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    # Benchmark multiple datasets
    all_results = []
    
    # Dataset 1: Breast Cancer
    print("\n" + "=" * 60)
    print("Dataset 1: Breast Cancer")
    data = load_breast_cancer()
    X_bc = pd.DataFrame(data.data, columns=data.feature_names)
    y_bc = data.target
    results_bc = benchmark_dataset("Breast Cancer", X_bc, y_bc, classifiers)
    all_results.append(results_bc)
    
    # Dataset 2: Wine
    print("\n" + "=" * 60)
    print("Dataset 2: Wine")
    data = load_wine()
    X_wine = pd.DataFrame(data.data, columns=data.feature_names)
    y_wine = data.target
    results_wine = benchmark_dataset("Wine", X_wine, y_wine, classifiers)
    all_results.append(results_wine)
    
    # Dataset 3: Iris
    print("\n" + "=" * 60)
    print("Dataset 3: Iris")
    data = load_iris()
    X_iris = pd.DataFrame(data.data, columns=data.feature_names)
    y_iris = data.target
    results_iris = benchmark_dataset("Iris", X_iris, y_iris, classifiers)
    all_results.append(results_iris)
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print("\nAccuracy Comparison:")
    print(final_results[['Dataset', 'Classifier', 'Baseline_Accuracy', 'FE_Accuracy', 'Accuracy_Improvement']].to_string(index=False))
    
    print("\n\nF1-Score Comparison:")
    print(final_results[['Dataset', 'Classifier', 'Baseline_F1', 'FE_F1', 'F1_Improvement']].to_string(index=False))
    
    print("\n\nFeature Count:")
    print(final_results[['Dataset', 'Classifier', 'Original_Features', 'Engineered_Features']].to_string(index=False))
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Average accuracy improvement: {final_results['Accuracy_Improvement'].mean():.2f}%")
    print(f"Average F1-score improvement: {final_results['F1_Improvement'].mean():.2f}%")
    print(f"Cases with improvement: {(final_results['Accuracy_Improvement'] > 0).sum()}/{len(final_results)}")
    print(f"Average feature count: {final_results['Original_Features'].mean():.0f} -> {final_results['Engineered_Features'].mean():.0f}")
    
    print("\n" + "=" * 60)
    print("Benchmark completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
