# Web Interface Preview

## Main Interface

The web application provides a clean, modern interface with the following sections:

### Header
- Purple gradient background with title "🚀 Automated Feature Engineering Engine"
- Subtitle: "Generate, score, and select the best features for your machine learning models"

### Section 1: Upload Dataset
- **CSV File Upload**: File input to select your dataset
- **Target Column**: Text input to specify the target variable name
- **Task Type**: Dropdown to select Classification or Regression

### Section 2: Configure Feature Engineering
- **Transformations**: Checkboxes for:
  - ✓ Polynomial Features (checked by default)
  - ✓ Binning (checked by default)
  - ☐ Target Encoding (unchecked by default)
  
- **Parameters**:
  - Polynomial Degree: Number input (default: 2)
  - Number of Bins: Number input (default: 5)
  - Number of Features to Select: Number input (default: 20)
  - Scoring Method: Dropdown (Mutual Information / Correlation)

### Action Buttons
- **Generate Features** (Purple gradient button)
- **Download Sample Data** (Gray button)
- **Reset** (Gray button)

### Results Section (appears after processing)
Shows 4 statistics cards:
1. **Original Features** - Number of features in uploaded data
2. **Generated Features** - Total features after transformation
3. **Selected Features** - Top-k features selected
4. **Samples** - Number of rows in dataset

Below the stats, a table displays:
- **Top Selected Features** table with columns:
  - Rank
  - Feature Name
  - Score (4 decimal places)

Processing time is shown at the bottom.

## Color Scheme
- Primary gradient: Purple (#667eea) to (#764ba2)
- Background: White cards on gradient background
- Accents: Light blue info boxes
- Tables: Purple headers with white rows

## Features
- Responsive design
- Loading spinner during processing
- Error messages in red boxes
- Clean, professional appearance
- Easy to use interface
