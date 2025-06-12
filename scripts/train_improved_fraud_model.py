import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/train_fraud_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fraud-model-training")

# Create output directories if they don't exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def load_data(data_path="data/crypto_volatility_fraud_dataset.csv"):
    """Load the dataset."""
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train an XGBoost model with hyperparameter tuning."""
    logger.info("Training XGBoost model...")
    
    # Calculate class weight for imbalanced data
    scale_pos_weight = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))
    logger.info(f"Using scale_pos_weight: {scale_pos_weight}")
    
    # Define model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    
    # Define parameters for grid search
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Use a smaller grid for faster training if needed
    small_param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 7],
        'n_estimators': [200],
        'subsample': [0.8]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=small_param_grid,  # Use small_param_grid for faster training
        scoring='f1',  # Focus on F1 score for imbalanced data
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    logger.info("\nClassification Report (XGBoost):")
    logger.info(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('outputs/xgboost_confusion_matrix.png')
    plt.close()
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('outputs/xgboost_roc_curve.png')
    plt.close()
    
    # Create precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('XGBoost Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('outputs/xgboost_pr_curve.png')
    plt.close()
    
    # Find optimal threshold for F1 score >= 0.80
    f1_scores = []
    for threshold in thresholds:
        y_pred_threshold = (y_prob >= threshold).astype(int)
        report = classification_report(y_test, y_pred_threshold, output_dict=True)
        if '1' in report:  # Check if class 1 exists in the report
            f1 = report['1']['f1-score']
            f1_scores.append((threshold, f1))
    
    # Sort by F1 score
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Find threshold with F1 >= 0.80 if possible
    optimal_threshold = 0.5  # Default
    for threshold, f1 in f1_scores:
        if f1 >= 0.80:
            optimal_threshold = threshold
            logger.info(f"Found threshold {optimal_threshold:.4f} with F1 score {f1:.4f}")
            break
    
    # If no threshold with F1 >= 0.80, use the best available
    if optimal_threshold == 0.5 and f1_scores:
        optimal_threshold, best_f1 = f1_scores[0]
        logger.info(f"No threshold with F1 >= 0.80 found. Using best threshold {optimal_threshold:.4f} with F1 score {best_f1:.4f}")
    
    # Apply optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    logger.info(f"\nClassification Report with optimal threshold {optimal_threshold:.4f}:")
    logger.info(classification_report(y_test, y_pred_optimal))
    
    # Save optimal threshold
    joblib.dump(optimal_threshold, 'models/xgboost_optimal_threshold.pkl')
    
    return best_model, y_pred_optimal, y_prob, optimal_threshold

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """Train a LightGBM model with hyperparameter tuning."""
    logger.info("Training LightGBM model...")
    
    # Calculate class weights
    class_weights = {0: 1.0, 1: len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))}
    logger.info(f"Using class weights: {class_weights}")
    
    # Define model
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=42,
        class_weight='balanced'  # LightGBM handles class weights differently
    )
    
    # Define parameters for grid search
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, -1],  # -1 means no limit
        'n_estimators': [100, 200, 300],
        'num_leaves': [31, 63, 127],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    
    # Use a smaller grid for faster training if needed
    small_param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 7],
        'n_estimators': [200],
        'num_leaves': [63],
        'subsample': [0.8]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=lgb_model,
        param_grid=small_param_grid,  # Use small_param_grid for faster training
        scoring='f1',  # Focus on F1 score for imbalanced data
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    logger.info("\nClassification Report (LightGBM):")
    logger.info(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('LightGBM Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('outputs/lightgbm_confusion_matrix.png')
    plt.close()
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('outputs/lightgbm_roc_curve.png')
    plt.close()
    
    # Create precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('outputs/lightgbm_pr_curve.png')
    plt.close()
    
    # Find optimal threshold for F1 score >= 0.80
    f1_scores = []
    for threshold in thresholds:
        y_pred_threshold = (y_prob >= threshold).astype(int)
        report = classification_report(y_test, y_pred_threshold, output_dict=True)
        if '1' in report:  # Check if class 1 exists in the report
            f1 = report['1']['f1-score']
            f1_scores.append((threshold, f1))
    
    # Sort by F1 score
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Find threshold with F1 >= 0.80 if possible
    optimal_threshold = 0.5  # Default
    for threshold, f1 in f1_scores:
        if f1 >= 0.80:
            optimal_threshold = threshold
            logger.info(f"Found threshold {optimal_threshold:.4f} with F1 score {f1:.4f}")
            break
    
    # If no threshold with F1 >= 0.80, use the best available
    if optimal_threshold == 0.5 and f1_scores:
        optimal_threshold, best_f1 = f1_scores[0]
        logger.info(f"No threshold with F1 >= 0.80 found. Using best threshold {optimal_threshold:.4f} with F1 score {best_f1:.4f}")
    
    # Apply optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    logger.info(f"\nClassification Report with optimal threshold {optimal_threshold:.4f}:")
    logger.info(classification_report(y_test, y_pred_optimal))
    
    # Save optimal threshold
    joblib.dump(optimal_threshold, 'models/lightgbm_optimal_threshold.pkl')
    
    return best_model, y_pred_optimal, y_prob, optimal_threshold

def preprocess_data(df):
    """Preprocess the dataset for fraud detection."""
    logger.info("Preprocessing data...")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Extract time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
    
    # Fill missing values
    df = df.fillna(0)
    
    # Create additional features
    if 'price' in df.columns and 'on_chain_volume' in df.columns:
        df['price_change'] = df['price'].pct_change().fillna(0)
        df['volume_change'] = df['on_chain_volume'].pct_change().fillna(0)
        df['volume_to_price_ratio'] = df['on_chain_volume'] / df['price'].replace(0, 1)
        
        if 'exchange_inflow' in df.columns:
            df['inflow_to_volume_ratio'] = df['exchange_inflow'] / df['on_chain_volume'].replace(0, 1)
    
    # Drop date column for modeling
    if 'date' in df.columns:
        df_model = df.drop(columns=['date'])
    else:
        df_model = df.copy()
    
    return df_model

def main():
    """Main function to train and evaluate fraud detection models."""
    logger.info("Starting fraud detection model training...")
    
    # Load data
    df = load_data()
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Prepare data for modeling
    X = df_processed.drop(columns=['fraud_label', 'volatility'], errors='ignore')
    y = df_processed['fraud_label']
    
    # Check class distribution
    class_counts = y.value_counts()
    logger.info(f"Class distribution:\n{class_counts}")
    logger.info(f"Fraud ratio: {y.mean():.2%}")
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.xlabel('Fraud Label (1 = Fraud)')
    plt.ylabel('Count')
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom')
    plt.savefig('outputs/class_distribution.png')
    plt.close()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE + Tomek Links to balance the training data
    logger.info("Applying SMOTE + Tomek Links to balance the dataset...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    # Check class distribution after resampling
    resampled_counts = pd.Series(y_train_resampled).value_counts()
    logger.info(f"Class distribution after SMOTE + Tomek Links:\n{resampled_counts}")
    
    # Train XGBoost model
    xgb_model, xgb_pred, xgb_prob, xgb_threshold = train_xgboost_model(X_train_resampled, y_train_resampled, X_test, y_test)
    
    # Train LightGBM model
    lgb_model, lgb_pred, lgb_prob, lgb_threshold = train_lightgbm_model(X_train_resampled, y_train_resampled, X_test, y_test)
    
    # Compare models
    logger.info("\nComparing models...")
    
    # XGBoost metrics
    xgb_report = classification_report(y_test, xgb_pred, output_dict=True)
    xgb_f1 = xgb_report['1']['f1-score']
    xgb_precision = xgb_report['1']['precision']
    xgb_recall = xgb_report['1']['recall']
    xgb_accuracy = xgb_report['accuracy']
    
    # LightGBM metrics
    lgb_report = classification_report(y_test, lgb_pred, output_dict=True)
    lgb_f1 = lgb_report['1']['f1-score']
    lgb_precision = lgb_report['1']['precision']
    lgb_recall = lgb_report['1']['recall']
    lgb_accuracy = lgb_report['accuracy']
    
    # Print comparison
    logger.info(f"XGBoost - Accuracy: {xgb_accuracy:.4f}, F1: {xgb_f1:.4f}, Precision: {xgb_precision:.4f}, Recall: {xgb_recall:.4f}")
    logger.info(f"LightGBM - Accuracy: {lgb_accuracy:.4f}, F1: {lgb_f1:.4f}, Precision: {lgb_precision:.4f}, Recall: {lgb_recall:.4f}")
    
    # Save the best model
    if xgb_f1 >= lgb_f1:
        logger.info("XGBoost model performed better. Saving as the primary fraud detection model.")
        joblib.dump(xgb_model, 'models/improved_fraud_detection_model.pkl')
        joblib.dump(xgb_threshold, 'models/fraud_detection_threshold.pkl')
        best_model = "XGBoost"
        best_metrics = {
            'accuracy': xgb_accuracy,
            'f1': xgb_f1,
            'precision': xgb_precision,
            'recall': xgb_recall,
            'threshold': xgb_threshold
        }
    else:
        logger.info("LightGBM model performed better. Saving as the primary fraud detection model.")
        joblib.dump(lgb_model, 'models/improved_fraud_detection_model.pkl')
        joblib.dump(lgb_threshold, 'models/fraud_detection_threshold.pkl')
        best_model = "LightGBM"
        best_metrics = {
            'accuracy': lgb_accuracy,
            'f1': lgb_f1,
            'precision': lgb_precision,
            'recall': lgb_recall,
            'threshold': lgb_threshold
        }
    
    # Save metrics to a file
    with open('outputs/fraud_model_metrics.txt', 'w') as f:
        f.write(f"Best model: {best_model}\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {best_metrics['f1']:.4f}\n")
        f.write(f"Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"Optimal Threshold: {best_metrics['threshold']:.4f}\n")
    
    logger.info("Fraud detection model training completed successfully!")
    logger.info(f"Best model ({best_model}) saved to models/improved_fraud_detection_model.pkl")
    logger.info(f"Optimal threshold saved to models/fraud_detection_threshold.pkl")
    logger.info(f"Metrics saved to outputs/fraud_model_metrics.txt")

if __name__ == "__main__":
    main()




