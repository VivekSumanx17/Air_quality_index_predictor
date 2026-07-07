import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def train_regression_model(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 60)
    print("AQI REGRESSION MODEL (XGBoost)")
    print("=" * 60)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric='mae'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test)
    }

    print(f"\nModel Performance:")
    print(f" Training MAE: {results['train_mae']:.2f}")
    print(f" Test MAE: {results['test_mae']:.2f}")
    print(f" Test RMSE: {results['test_rmse']:.2f}")
    print(f" Test R² Score: {results['test_r2']:.4f}")

    return model, results, y_pred_test


def train_classification_model(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 60)
    print("AQI CATEGORY CLASSIFICATION (XGBoost)")
    print("=" * 60)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric='mlogloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred_test = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_test)

    print(f"\nModel Performance:")
    print(f" Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    return model, accuracy, y_pred_test


def plot_predictions(y_true, y_pred, title="AQI Predictions"):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual AQI')
    axes[0].set_ylabel('Predicted AQI')
    axes[0].set_title(f'{title}\nScatter Plot')

    axes[1].plot(y_true.values[:100], label='Actual', alpha=0.7)
    axes[1].plot(y_pred[:100], label='Predicted', alpha=0.7)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('AQI')
    axes[1].set_title('Actual vs Predicted (First 100 points)')
    axes[1].legend()

    residuals = y_true - y_pred
    axes[2].scatter(y_pred, residuals, alpha=0.5)
    axes[2].axhline(y=0, color='r', linestyle='--')
    axes[2].set_xlabel('Predicted AQI')
    axes[2].set_ylabel('Residuals')
    axes[2].set_title('Residual Plot')

    plt.tight_layout()
    plt.savefig('models/predictions_plot.png', dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150)
    plt.show()
    print("Saved feature importance plot to models/feature_importance.png")


def plot_training_history(model):
    if hasattr(model, 'evals_result'):
        results = model.evals_result()
        plt.figure(figsize=(10, 5))
        plt.plot(results['validation_0']['mae'], label='Validation MAE')
        plt.xlabel('Boosting Rounds')
        plt.ylabel('MAE')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('models/training_history.png', dpi=150)
        plt.show()
        print("Saved training history to models/training_history.png")


def calculate_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Satisfactory"
    elif aqi_value <= 200:
        return "Moderate"
    elif aqi_value <= 300:
        return "Poor"
    elif aqi_value <= 400:
        return "Very Poor"
    else:
        return "Severe"


def evaluate_categorical_predictions(y_true, y_pred):
    results = {}
    for category in ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']:
        mask = y_true == category
        if mask.sum() > 0:
            accuracy = (y_pred[mask] == category).sum() / mask.sum()
            results[category] = accuracy

    print("\nCategory-wise Accuracy:")
    for cat, acc in results.items():
        print(f"   {cat}: {acc:.4f}")

    return results


# Main execution
if __name__ == "__main__":
    # Load prepared data
    data = joblib.load("dataset/prepared_data.pkl")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = data

    print("Starting Model Training...")
    print(f"Training samples: {X_train.shape}")
    print(f"Features: {X_train.shape[1]}")
    print(f"AQI Range: {y_train.min():.1f} - {y_train.max():.1f}")

    # Train model
    model, results, y_pred = train_regression_model(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Plot results
    plot_predictions(y_test, y_pred, "AQI Predictions - Test Set")
    plot_feature_importance(model, feature_cols, top_n=20)

    # Save model and metadata
    joblib.dump(model, "models/aqi_regression_model.pkl")

    metadata = {
        'features': feature_cols,
        'results': results,
        'model_type': 'XGBoost_Regression'
    }
    joblib.dump(metadata, "models/model_metadata.pkl")

    print("\nTraining complete!")

    sample_idx = 0
    sample = X_test.iloc[sample_idx:sample_idx + 1]
    prediction = model.predict(sample)[0]
    actual = y_test.iloc[sample_idx]

    print("\nSample Prediction:")
    print(f" Actual AQI: {actual:.1f} ({calculate_aqi_category(actual)})")
    print(f" Predicted AQI: {prediction:.1f} ({calculate_aqi_category(prediction)})")
    print(f" Error: {abs(actual - prediction):.1f} points")

    try:
        df_features = pd.read_csv("dataset/features_data.csv")
        # if 'AQI_Bucket_encoded' in df_features.columns:
        print("\nAlso training classification model...")
        y_train_class = df_features.loc[y_train.index, 'AQI_Bucket_encoded'] if hasattr(y_train, 'index') else None
        if y_train_class is not None:
            class_model, class_acc, class_pred = train_classification_model(
                X_train, y_train_class, X_val, y_train_class[:len(X_val)],
                X_test, y_train_class[:len(X_test)]
            )
            joblib.dump(class_model, "models/aqi_classification_model.pkl")
    except:
        print("\nClassification model not trained")