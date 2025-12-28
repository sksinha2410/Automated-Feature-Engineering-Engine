"""
Flask web application for testing the Automated Feature Engineering Engine.

This web app provides an interactive interface to upload datasets and
test the feature engineering capabilities.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io
import time
from sklearn.datasets import load_breast_cancer
import sys
import os

# Add parent directory to path to import feature_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engine import AutoFeatureEngineeringPipeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Process uploaded CSV file with feature engineering."""
    try:
        start_time = time.time()
        
        # Get file and parameters
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read parameters
        target_column = request.form.get('target_column', '')
        task_type = request.form.get('task_type', 'classification')
        use_polynomial = request.form.get('use_polynomial', 'false').lower() == 'true'
        use_binning = request.form.get('use_binning', 'false').lower() == 'true'
        use_target_encoding = request.form.get('use_target_encoding', 'false').lower() == 'true'
        polynomial_degree = int(request.form.get('polynomial_degree', 2))
        n_bins = int(request.form.get('n_bins', 5))
        k_features = int(request.form.get('k_features', 20))
        scoring_method = request.form.get('scoring_method', 'mutual_info')
        
        # Read CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
        
        # Validate target column
        if target_column not in df.columns:
            return jsonify({
                'error': f'Target column "{target_column}" not found. Available columns: {", ".join(df.columns)}'
            }), 400
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        original_features = X.shape[1]
        
        # Create and fit pipeline
        pipeline = AutoFeatureEngineeringPipeline(
            use_polynomial=use_polynomial,
            use_binning=use_binning,
            use_target_encoding=use_target_encoding,
            polynomial_degree=polynomial_degree,
            n_bins=n_bins,
            k_features=min(k_features, original_features * 10),  # Reasonable limit
            scoring_method=scoring_method,
            task=task_type
        )
        
        # Transform data
        X_transformed = pipeline.fit_transform(X, y)
        
        # Get feature scores
        if hasattr(pipeline.pipeline_.named_steps.get('feature_selection', None), 'scorer_'):
            scorer = pipeline.pipeline_.named_steps['feature_selection'].scorer_
            scores_df = scorer.get_scores()
            
            # Get top features
            top_features = []
            for idx, row in scores_df.head(min(20, len(scores_df))).iterrows():
                top_features.append({
                    'name': row['feature'],
                    'score': float(row['score'])
                })
        else:
            top_features = [{'name': col, 'score': 0.0} for col in X_transformed.columns[:20]]
        
        processing_time = time.time() - start_time
        
        # Prepare response
        result = {
            'original_features': original_features,
            'generated_features': X_transformed.shape[1] if hasattr(pipeline.pipeline_.named_steps.get('feature_generation'), 'transform') else original_features,
            'selected_features': X_transformed.shape[1],
            'data_shape': {
                'samples': X.shape[0],
                'features': X_transformed.shape[1]
            },
            'top_features': top_features,
            'processing_time': processing_time
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error processing data: {str(e)}'}), 500


@app.route('/sample', methods=['GET'])
def sample():
    """Load sample breast cancer dataset."""
    try:
        # Load breast cancer dataset
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        # Save to CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=breast_cancer.csv'
        }
    except Exception as e:
        return jsonify({'error': f'Error loading sample data: {str(e)}'}), 500


if __name__ == '__main__':
    print('=' * 60)
    print('Automated Feature Engineering Engine - Web Demo')
    print('=' * 60)
    print('\nStarting server...')
    print('Open your browser and navigate to: http://127.0.0.1:5000')
    print('\nPress Ctrl+C to stop the server')
    print('=' * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
