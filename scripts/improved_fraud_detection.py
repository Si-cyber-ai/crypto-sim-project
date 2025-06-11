import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('../data/crypto_volatility_fraud_dataset.csv')

# Basic preprocessing
print("Preprocessing data...")
# Convert date to datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    # Extract time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    # Drop date column for modeling
    df_model = df.drop(columns=['date'])
else:
    df_model = df.copy()

# Fill missing values if any
df_model = df_model.fillna(0)

# Create additional features
df_model['price_change'] = df_model['price'].pct_change().fillna(0)
df_model['volume_change'] = df_model['on_chain_volume'].pct_change().fillna(0)
df_model['volume_to_price_ratio'] = df_model['on_chain_volume'] / df_model['price']
df_model['inflow_to_volume_ratio'] = df_model['exchange_inflow'] / df_model['on_chain_volume'].replace(0, 1)

# Prepare data for modeling
X = df_model.drop(columns=['fraud_label', 'volatility'])
y = df_model['fraud_label']

# Check class distribution
print("\nClass distribution before resampling:")
print(y.value_counts())
print(f"Fraud percentage: {y.mean() * 100:.2f}%")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models to try
models = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])),
    "Isolation Forest": IsolationForest(random_state=42, contamination=y_train.mean())
}

# Define parameters for grid search
param_grids = {
    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    "XGBoost": {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200]
    }
}

# Train and evaluate models
best_models = {}
best_f1 = 0
best_model_name = None

for name, model in models.items():
    print(f"\n{'-'*50}")
    print(f"Training {name} model...")
    
    if name == "Isolation Forest":
        # Isolation Forest doesn't need SMOTE and works differently
        model.fit(X_train)
        # Convert outlier scores to binary predictions (invert because -1 is outlier in Isolation Forest)
        y_pred = (model.predict(X_test) == -1).astype(int)
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'../models/{name.lower().replace(" ", "_")}_confusion_matrix.png')
        
        # Save model
        joblib.dump(model, f'../models/{name.lower().replace(" ", "_")}.pkl')
        best_models[name] = model
        
        # Calculate F1 score for fraud class
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
    else:
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # Grid search with cross-validation
        if name in param_grids:
            param_grid = {f'classifier__{key}': value for key, value in param_grids[name].items()}
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
            
            pipeline = grid_search.best_estimator_
        else:
            pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'../models/{name.lower().replace(" ", "_")}_confusion_matrix.png')
        
        # ROC Curve if probabilities are available
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{name} ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(f'../models/{name.lower().replace(" ", "_")}_roc_curve.png')
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{name} Precision-Recall Curve')
            plt.savefig(f'../models/{name.lower().replace(" ", "_")}_pr_curve.png')
        
        # Save model
        joblib.dump(pipeline, f'../models/{name.lower().replace(" ", "_")}_pipeline.pkl')
        best_models[name] = pipeline
        
        # Calculate F1 score for fraud class
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

# Save the best model as the default fraud detection model
if best_model_name:
    print(f"\nBest model: {best_model_name} with fraud F1-score: {best_f1:.4f}")
    joblib.dump(best_models[best_model_name], '../models/fraud_detection_model.pkl')
    print("✅ Best model saved as fraud_detection_model.pkl")

# Feature importance for the best model (if applicable)
if best_model_name in ["Random Forest", "XGBoost"]:
    model = best_models[best_model_name]
    
    if best_model_name == "Random Forest":
        # Get feature importance from the classifier in the pipeline
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        else:
            importances = model.feature_importances_
    elif best_model_name == "XGBoost":
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        else:
            importances = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.savefig(f'../models/feature_importance.png')
    print("✅ Feature importance plot saved")

print("\nFraud detection model training complete!")