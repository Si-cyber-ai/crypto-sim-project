import subprocess
import sys
import os

def install_dependencies():
    """Install all required dependencies for the project."""
    print("Installing dependencies...")
    
    # Core dependencies
    dependencies = [
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "streamlit",
        "plotly",
        "pywavelets",
        "joblib",
        "imbalanced-learn",
        "xgboost",
        "lightgbm"
    ]
    
    for package in dependencies:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n✅ All dependencies installed successfully!")
    print("\nYou can now run the following commands:")
    print("1. Generate data: python scripts/generate_crypto_data.py")
    print("2. Train models: python scripts/run_training.py")
    print("3. Run dashboard: python scripts/dashboard.py")
    print("4. Run web app: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    install_dependencies()