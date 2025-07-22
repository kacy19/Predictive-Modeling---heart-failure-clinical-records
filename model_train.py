# model_train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def load_data(filename):
    """Load the dataset with error handling for different file formats and names"""
    possible_filenames = [
        filename,
        "heart_failure_clinical_records_dataset(1).csv",
        "heart_failure_clinical_records_dataset.csv"
    ]
    
    for fname in possible_filenames:
        if os.path.exists(fname):
            try:
                # Try reading as CSV first
                df = pd.read_csv(fname)
                print(f"Successfully loaded {fname} as CSV")
                return df
            except Exception as e:
                try:
                    # If CSV fails, try Excel
                    df = pd.read_excel(fname)
                    print(f"Successfully loaded {fname} as Excel")
                    return df
                except Exception as e2:
                    print(f"Failed to load {fname}: {e2}")
                    continue
    
    raise FileNotFoundError(f"Could not find or load any of these files: {possible_filenames}")

def main():
    try:
        # Load the dataset
        df = load_data("heart_failure_clinical_records_dataset.csv")
        
        # Display basic info about the dataset
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Check if DEATH_EVENT column exists
        if "DEATH_EVENT" not in df.columns:
            print("Available columns:", list(df.columns))
            raise ValueError("DEATH_EVENT column not found in dataset")
        
        # Handle missing values if any
        if df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            df = df.fillna(df.mean(numeric_only=True))
        
        # Features and target
        X = df.drop("DEATH_EVENT", axis=1)
        y = df["DEATH_EVENT"]
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train a model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        # Save the model
        with open("model.pkl", "wb") as file:
            pickle.dump(model, file)
        
        # Also save feature names for future use
        with open("feature_names.pkl", "wb") as file:
            pickle.dump(list(X.columns), file)
        
        print("\nModel trained and saved as model.pkl")
        print("Feature names saved as feature_names.pkl")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the dataset file is in the same directory as this script.")
    except ValueError as e:
        print(f"Data Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
