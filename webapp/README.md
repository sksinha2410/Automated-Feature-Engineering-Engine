# Web Application for Feature Engineering Engine

This directory contains a Flask web application that provides an interactive interface to test the Automated Feature Engineering Engine.

## Features

- Upload CSV datasets
- Configure feature engineering parameters
- Interactive visualization of results
- Real-time processing
- Download sample datasets

## Installation

```bash
# Make sure the main library and its dependencies are installed first
cd ..
pip install -r requirements.txt
pip install -e .

# Then install web-specific dependencies
cd webapp
pip install -r requirements.txt
```

## Running the Web App

```bash
cd webapp
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. **Upload Dataset**: Select a CSV file with your tabular data
2. **Specify Target**: Enter the name of your target column
3. **Configure Parameters**:
   - Choose transformations (polynomial, binning, target encoding)
   - Set polynomial degree
   - Set number of bins
   - Specify how many features to select
   - Choose scoring method
4. **Generate Features**: Click the button to process your data
5. **View Results**: See statistics and top selected features

## Alternative: Download Sample Data

Click "Download Sample Data" to get the breast cancer dataset, then upload it to test the application.

## API Endpoints

- `GET /` - Main web interface
- `POST /process` - Process uploaded CSV file
- `GET /sample` - Download sample breast cancer dataset

## Screenshots

The web interface provides:
- Clean, modern UI with gradient design
- Real-time processing with loading indicator
- Statistics dashboard showing feature counts
- Top features table with scores
- Error handling and validation

## Notes

- Maximum file size: 16MB
- Supported formats: CSV
- Requires target column in the dataset
- Processing time depends on dataset size and parameters
