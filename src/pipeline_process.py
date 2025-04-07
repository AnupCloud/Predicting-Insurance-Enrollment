import warnings

from src.config import DATASET

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, \
    log_loss
import joblib
import os

# Create output directories
os.makedirs('output_logistic', exist_ok=True)


def create_learning_curves(model, X_train, y_train, X_val, y_val):
    """
    Generate learning curves for the logistic regression model

    Args:
        model: Trained model object
        X_train, y_train: Training data
        X_val, y_val: Validation data
    """
    # Set up figures for accuracy and loss
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))

    # Training sizes to evaluate (from 10% to 100% of training data)
    train_sizes = np.linspace(0.1, 1.0, 10)

    # Calculate learning curves for accuracy
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes,
        cv=5, scoring='accuracy',
        n_jobs=-1)

    # Calculate mean and std for training and validation scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Plot accuracy learning curve
    ax_acc.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training accuracy')
    ax_acc.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    ax_acc.plot(train_sizes_abs, val_mean, 'o-', color='g', label='Validation accuracy')
    ax_acc.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    ax_acc.set_title('Learning Curve - Accuracy')
    ax_acc.set_xlabel('Training Examples')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='best')
    ax_acc.grid(True)

    # Calculate log loss for different training sizes
    train_loss = np.zeros(len(train_sizes_abs))
    val_loss = np.zeros(len(train_sizes_abs))

    for j, train_size in enumerate(train_sizes_abs):
        # Subsample the training data
        X_train_subset, _, y_train_subset, _ = train_test_split(
            X_train, y_train, train_size=train_size / len(X_train), random_state=42)

        # Train the model
        model.fit(X_train_subset, y_train_subset)

        # Calculate log loss on training and validation sets
        train_prob = model.predict_proba(X_train_subset)
        val_prob = model.predict_proba(X_val)

        train_loss[j] = log_loss(y_train_subset, train_prob)
        val_loss[j] = log_loss(y_val, val_prob)

    # Plot loss learning curve
    ax_loss.plot(train_sizes_abs, train_loss, 'o-', color='r', label='Training loss')
    ax_loss.plot(train_sizes_abs, val_loss, 'o-', color='g', label='Validation loss')
    ax_loss.set_title('Learning Curve - Log Loss')
    ax_loss.set_xlabel('Training Examples')
    ax_loss.set_ylabel('Log Loss')
    ax_loss.legend(loc='best')
    ax_loss.grid(True)

    # Adjust layout and save figures
    fig_acc.tight_layout()
    fig_acc.savefig('output_logistic/accuracy_learning_curve.png')

    fig_loss.tight_layout()
    fig_loss.savefig('output_logistic/loss_learning_curve.png')

    plt.close('all')


def create_regularization_plot(model_results, metric_name='accuracy'):
    """
    Create a plot showing the effect of regularization strength

    Args:
        model_results: Dictionary with model metrics
        metric_name: Metric to plot (accuracy or loss)
    """
    plt.figure(figsize=(10, 6))

    # Extract data from results
    C_values = []
    train_metrics = []
    val_metrics = []

    for C, metrics in model_results.items():
        C_values.append(C)
        train_metrics.append(metrics[f'train_{metric_name}'])
        val_metrics.append(metrics[f'val_{metric_name}'])

    # Sort by C value
    sorted_data = sorted(zip(C_values, train_metrics, val_metrics))
    C_values, train_metrics, val_metrics = zip(*sorted_data)

    # Plot
    plt.plot(C_values, train_metrics, 'ro-', label='Training')
    plt.plot(C_values, val_metrics, 'go-', label='Validation')

    # Set axis labels and title
    plt.xlabel('Regularization Strength (C)')
    plt.xscale('log')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Effect of Regularization on {metric_name.capitalize()}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Save figure
    plt.tight_layout()
    plt.savefig(f'output_logistic/regularization_{metric_name}.png')
    plt.close()


def visualize_feature_importance(model, feature_names):
    """
    Visualize feature coefficients for logistic regression

    Args:
        model: Trained logistic regression model
        feature_names: Names of the features
    """
    # Get coefficients
    coefs = model.coef_[0]

    # Create DataFrame for visualization
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    })

    # Sort by absolute coefficient value
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot bars with positive/negative colors
    bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'])

    # Color bars based on coefficient sign
    for i, bar in enumerate(bars):
        if coef_df['Coefficient'].iloc[i] < 0:
            bar.set_color('r')
        else:
            bar.set_color('g')

    plt.title('Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig('output_logistic/feature_coefficients.png')
    plt.close()


def logistic_regression_pipeline(data_path):
    """
    Run a pipeline to create a logistic regression model
    """
    print("\n=== Insurance Enrollment Model with Logistic Regression ===\n")

    # Step 1: Load the data
    print("Step 1: Loading data...")
    data = pd.read_csv(data_path)
    print(f"Data loaded successfully with shape: {data.shape}")

    # Step 2: Basic data exploration
    print("\nStep 2: Basic data exploration...")
    print("Target distribution:")
    print(data['enrolled'].value_counts(normalize=True))

    # Step 3: Feature Engineering and Preprocessing
    print("\nStep 3: Feature engineering and preprocessing...")

    # Create a copy of the dataset
    data_processed = data.copy()

    # Identify column types
    id_cols = ['employee_id']
    numeric_cols = ['age', 'salary', 'tenure_years']
    categorical_cols = ['gender', 'marital_status', 'employment_type', 'region', 'has_dependents']
    target_col = 'enrolled'

    # Handle categorical variables - convert to proper format
    data_processed['has_dependents'] = (data_processed['has_dependents'] == 'Yes').astype(int)

    # Create feature transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop other columns
    )

    # Define X and y
    X = data_processed.drop(columns=id_cols + [target_col])
    y = data_processed[target_col]

    # Get the feature names for the transformed data
    categorical_features = []
    for col in categorical_cols:
        unique_values = data_processed[col].unique()
        # We drop the first category in OneHotEncoder
        categorical_features.extend([f"{col}_{val}" for val in unique_values[1:]])

    all_features = numeric_cols + categorical_features

    # Step 4: Create three separate datasets
    print("\nStep 4: Creating training, validation, and test sets...")

    # First split off test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

    # Then split remaining data into train and validation (validation is 25% of original)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3125,
                                                      random_state=42, stratify=y_temp)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Step 5: Preprocess the data
    print("\nStep 5: Preprocessing the data...")

    # Fit the preprocessor and transform the data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)

    print(f"Preprocessed training data shape: {X_train_preprocessed.shape}")

    # Step 6: Apply class balancing
    print("\nStep 6: Applying class balancing...")

    # Separate majority and minority classes
    majority_indices = np.where(y_train == 1)[0]
    minority_indices = np.where(y_train == 0)[0]

    # Downsample majority class to be slightly smaller than minority
    minority_size = len(minority_indices)
    downsampled_size = int(minority_size * 0.9)  # 90% of minority size

    # Randomly select indices from majority class
    downsampled_indices = np.random.choice(majority_indices, size=downsampled_size, replace=False)

    # Combine minority and downsampled majority indices
    balanced_indices = np.concatenate([downsampled_indices, minority_indices])

    # Create balanced datasets
    X_train_balanced = X_train_preprocessed[balanced_indices]
    y_train_balanced = y_train.iloc[balanced_indices].reset_index(drop=True)

    print(f"Original training class distribution: {y_train.value_counts(normalize=True)}")
    print(f"Balanced training class distribution: {y_train_balanced.value_counts(normalize=True)}")

    # Step 7: Add random noise to improve robustness
    print("\nStep 7: Adding random noise to training data...")

    # Create a copy to avoid modifying original data
    X_train_noisy = X_train_balanced.copy()

    # Add noise to numerical features only
    np.random.seed(42)
    noise_level = 0.2

    # The first few columns are the numeric features
    num_numeric = len(numeric_cols)
    for i in range(num_numeric):
        noise = np.random.normal(0, noise_level, size=X_train_noisy.shape[0])
        X_train_noisy[:, i] = X_train_noisy[:, i] + noise

    # Step 8: Train logistic regression models with different regularization strengths
    print("\nStep 8: Training logistic regression models with different regularization strengths...")

    # Try different C values
    C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

    log_reg_models = {}
    model_results = {}

    for C in C_values:
        print(f"Regularization strength C={C}...")

        # Create a logistic regression model with L2 regularization
        log_reg = LogisticRegression(
            C=C,
            penalty='l2',
            solver='liblinear',
            random_state=42,
            max_iter=2000
        )

        # Train the model
        log_reg.fit(X_train_noisy, y_train_balanced)

        # Calculate metrics
        train_acc = log_reg.score(X_train_balanced, y_train_balanced)  # Use balanced but not noisy data
        val_acc = log_reg.score(X_val_preprocessed, y_val)

        train_probs = log_reg.predict_proba(X_train_balanced)
        val_probs = log_reg.predict_proba(X_val_preprocessed)
        train_loss = log_loss(y_train_balanced, train_probs)
        val_loss = log_loss(y_val, val_probs)

        # Store model and results
        model_name = f"Logistic Regression (C={C})"
        log_reg_models[model_name] = log_reg
        model_results[C] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        print(f"Training accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")
        print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

    # Step 9: Create visualizations
    print("\nStep 9: Creating visualizations...")

    # Get feature names for transformed data
    feature_names = []
    feature_names.extend(numeric_cols)  # Add numeric column names

    # Add one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    # Get the categories for each categorical column
    for i, col in enumerate(categorical_cols):
        categories = ohe.categories_[i][1:]  # Skip first category (dropped)
        feature_names.extend([f"{col}_{cat}" for cat in categories])

    # Create regularization plots
    create_regularization_plot(model_results, 'accuracy')
    create_regularization_plot(model_results, 'loss')

    # Step 10: Select the best model
    print("\nStep 10: Selecting the best model...")

    # Find best model based on validation accuracy
    best_C = max(model_results, key=lambda C: model_results[C]['val_accuracy'])
    best_model = log_reg_models[f"Logistic Regression (C={best_C})"]
    best_val_acc = model_results[best_C]['val_accuracy']

    print(f"Best model: C={best_C}, Validation accuracy: {best_val_acc:.4f}")
    train_val_gap = model_results[best_C]['train_accuracy'] - best_val_acc
    print(f"Training-validation accuracy gap: {train_val_gap:.4f}")

    # Create learning curves for the best model
    create_learning_curves(best_model, X_train_balanced, y_train_balanced, X_val_preprocessed, y_val)

    # Visualize feature importance
    visualize_feature_importance(best_model, feature_names)

    # Step 11: Evaluate on test set
    print("\nStep 11: Evaluating the best model on test set...")

    # Make predictions
    y_pred = best_model.predict(X_test_preprocessed)
    y_prob = best_model.predict_proba(X_test_preprocessed)[:, 1]

    # Calculate metrics
    test_accuracy = (y_pred == y_test).mean()
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test ROC AUC: {test_auc:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step 12: Final visualizations
    print("\nStep 12: Creating final visualizations...")

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('output_logistic/confusion_matrix.png')
    plt.close()

    # ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {test_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig('output_logistic/roc_curve.png')
    plt.close()

    # Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Logistic Regression')
    plt.legend(loc="best")
    plt.savefig('output_logistic/precision_recall_curve.png')
    plt.close()

    # Step 13: Save the model
    print("\nStep 13: Saving the model...")

    model_info = {
        'model': best_model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'accuracy': test_accuracy,
        'roc_auc': test_auc
    }

    model_path = 'output_logistic/final_model.pkl'
    joblib.dump(model_info, model_path)
    print(f"Model saved to {model_path}")

    # Step 14: Create prediction function
    print("\nStep 14: Creating prediction function...")

    def predict_enrollment(new_data):
        """
        Predict enrollment probability for new employees

        Args:
            new_data (pandas.DataFrame): Employee data

        Returns:
            pandas.DataFrame: Original data with enrollment probabilities
        """
        # Create a copy of the input data
        result = new_data.copy()

        # Preprocess the data using the same preprocessor
        processed_data = preprocessor.transform(new_data)

        # Make predictions
        enrollment_prob = best_model.predict_proba(processed_data)[:, 1]

        # Add predictions to the original data
        result['enrollment_probability'] = enrollment_prob
        result['predicted_enrollment'] = (enrollment_prob >= 0.5).astype(int)

        return result

    # Step 15: Test the prediction function
    print("\nStep 15: Testing prediction function with example data...")

    # Create example data
    example_data = pd.DataFrame({
        'employee_id': [20001, 20002, 20003, 20004],
        'age': [35, 50, 42, 29],
        'gender': ['Male', 'Female', 'Other', 'Female'],
        'marital_status': ['Married', 'Single', 'Divorced', 'Single'],
        'salary': [75000, 85000, 62000, 45000],
        'employment_type': ['Full-time', 'Part-time', 'Full-time', 'Contract'],
        'region': ['West', 'Northeast', 'South', 'Midwest'],
        'has_dependents': ['Yes', 'No', 'Yes', 'No'],
        'tenure_years': [5.5, 2.0, 8.3, 1.2]
    })

    # Make predictions
    example_predictions = predict_enrollment(example_data)

    # Display results
    print("\nExample prediction results:")
    print(example_predictions[['employee_id', 'employment_type', 'has_dependents',
                               'salary', 'enrollment_probability', 'predicted_enrollment']])

    # Step 16: Feature analysis
    print("\nStep 16: Feature importance analysis...")

    # Get coefficients from the model
    coefs = best_model.coef_[0]

    # Create DataFrame with feature names and coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Absolute_Value': np.abs(coefs)
    })

    # Sort by absolute coefficient value
    coef_df = coef_df.sort_values('Absolute_Value', ascending=False)

    # Print top positive features
    print("\nTop positive factors for enrollment:")
    pos_features = coef_df[coef_df['Coefficient'] > 0].head(5)
    for idx, row in pos_features.iterrows():
        print(f"  {row['Feature']}: +{row['Coefficient']:.4f}")

    # Print top negative features
    print("\nTop negative factors for enrollment:")
    neg_features = coef_df[coef_df['Coefficient'] < 0].head(5)
    for idx, row in neg_features.iterrows():
        print(f"  {row['Feature']}: {row['Coefficient']:.4f}")

    # Step 17: Summarize the process
    print("\n=== Pipeline Summary ===")
    pipeline_summary = {
        'Model': f"Logistic Regression (C={best_C})",
        'Feature Count': len(feature_names),
        'Class Balancing': 'Applied (aggressive downsampling)',
        'Training Set Size': X_train.shape[0],
        'Validation Set Size': X_val.shape[0],
        'Test Set Size': X_test.shape[0],
        'Test Accuracy': test_accuracy,
        'Test ROC AUC': test_auc
    }

    for key, value in pipeline_summary.items():
        print(f"{key}: {value}")

    print("\nLogistic regression model pipeline completed successfully!")

    return predict_enrollment


# Main execution
if __name__ == "__main__":
    # Run the pipeline
    data_path = DATASET  # Replace with your data path
    predict_function = logistic_regression_pipeline(data_path)