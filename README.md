# Cryptocurrency Fraud Detection & Volatility Prediction

This project implements a machine learning pipeline for cryptocurrency fraud detection and volatility prediction using advanced techniques including wavelet decomposition, random forests, and transformer models.

## Features

- **Fraud Detection**: Identifies potentially fraudulent transactions using a Random Forest classifier
- **Volatility Prediction**: Forecasts cryptocurrency price volatility using a Transformer model
- **Interactive Dashboard**: Visualizes data, predictions, and model performance
- **Synthetic Data Generation**: Creates realistic cryptocurrency transaction data for testing

## Project Structure

```
crypto-ml-project/
├── app/
│   └── streamlit_app.py       # Interactive web application
├── data/
│   └── crypto_volatility_fraud_dataset.csv  # Generated dataset
├── models/                    # Saved models directory
├── logs/                      # Log files directory
├── scripts/
│   ├── crypto_ml_pipeline.py  # Main ML pipeline
│   ├── dashboard.py           # Data visualization dashboard
│   ├── generate_crypto_data.py # Synthetic data generator
│   └── run_training.py        # Script to run the training pipeline
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crypto-ml-project.git
   cd crypto-ml-project
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install torch torchvision
   pip install pandas numpy matplotlib seaborn scikit-learn
   pip install streamlit plotly
   pip install pywavelets joblib imbalanced-learn
   ```

   Or install all at once:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Synthetic Data

Generate synthetic cryptocurrency transaction data for testing:

```
python scripts/generate_crypto_data.py --samples 5000 --output data/crypto_volatility_fraud_dataset.csv
```

Options:
- `--samples`: Number of data points to generate (default: 5000)
- `--output`: Output file path (default: ../data/crypto_volatility_fraud_dataset.csv)
- `--fraud_ratio`: Ratio of fraudulent transactions (default: 0.05)

### 2. Train Models

Train the fraud detection and volatility prediction models:

```
python scripts/run_training.py
```

Or use the full pipeline with more options:

```
python scripts/crypto_ml_pipeline.py --mode production --data_path data/crypto_volatility_fraud_dataset.csv --balancing smote
```

Options:
- `--mode`: Running mode: debug or production (default: production)
- `--data_path`: Path to the dataset (default: ../data/crypto_volatility_fraud_dataset.csv)
- `--output_dir`: Directory to save models (default: ../models)
- `--balancing`: Class balancing method: smote, ros, or none (default: smote)
- `--volatility_model`: Model type for volatility prediction: transformer, lstm, or none (default: transformer)
- `--seq_length`: Sequence length for time series models (default: 5)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of epochs for training (default: 100)

### 3. Run the Dashboard

Launch the data visualization dashboard:

```
python scripts/dashboard.py
```

Options:
- `--data_path`: Path to the dataset (default: ../data/crypto_volatility_fraud_dataset.csv)

### 4. Run the Web Application

Launch the interactive Streamlit web application:

```
cd app
streamlit run streamlit_app.py
```

Or from the project root:

```
streamlit run app/streamlit_app.py
```

## Example Workflow

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate synthetic data
python scripts/generate_crypto_data.py --samples 10000

# 3. Train models
python scripts/run_training.py

# 4. Run dashboard
python scripts/dashboard.py

# 5. Run web application
streamlit run app/streamlit_app.py
```

## Requirements File

Create a `requirements.txt` file with the following content:

```
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
streamlit>=1.0.0
plotly>=5.0.0
pywavelets>=1.1.0
joblib>=1.0.0
imbalanced-learn>=0.8.0
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
