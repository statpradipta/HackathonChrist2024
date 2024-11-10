import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
    def train(self):
        # Load the real dataset
        try:
            df = pd.read_csv('general_disease_diagnosis.csv')
            print("Dataset loaded successfully!")
            print(f"Shape of dataset: {df.shape}")
            print("\nColumns in your dataset:")
            print(df.columns.tolist())
            
            # Display first few rows to understand structure
            print("\nFirst few rows of your dataset:")
            print(df.head())
            
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            print("\nNumeric columns in your dataset:")
            print(numeric_columns.tolist())
            
            # Use all numeric columns as features except the target column
            features = numeric_columns.tolist()
            target_column = None
            
            # Try to identify target column
            for col in df.columns:
                if 'condition' in col.lower() or 'disease' in col.lower() or 'diagnosis' in col.lower():
                    target_column = col
                    if col in features:
                        features.remove(col)
            
            if not target_column:
                print("\nWarning: Could not automatically identify target column.")
                print("Available columns:", df.columns.tolist())
                target_column = input("Please enter the name of your target column: ")
            
            print("\nUsing these features:", features)
            print("Target column:", target_column)
            
        except FileNotFoundError:
            print("Error: The file 'general_disease_diagnosis.csv' was not found in the current directory.")
            return
        except Exception as e:
            print(f"Error loading the dataset: {str(e)}")
            return
        
        # Handle missing values if any
        if df.isnull().sum().any():
            print("\nHandling missing values...")
            for column in df.columns:
                if df[column].dtype in ['float64', 'int64']:
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)
        
        # Prepare features and target
        X = df[features]
        y = df[target_column]
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature names for prediction
        self.feature_names = features
        
        # Train model
        print("\nTraining the model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate and print metrics
        accuracy = np.mean(y_pred == y_test)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        })
        print("\nFeature Importance:")
        print(feature_importance.sort_values('importance', ascending=False))
    
    def predict(self, *feature_values):
        """
        Make prediction using the trained model
        """
        if not hasattr(self, 'feature_names'):
            raise ValueError("Model hasn't been trained yet!")
            
        # Prepare input data
        input_data = np.array([feature_values])
        
        # Scale input
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction_encoded = self.model.predict(input_scaled)
        prediction = self.label_encoder.inverse_transform(prediction_encoded)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(input_scaled)[0]
        conditions_with_probs = list(zip(
            self.label_encoder.classes_,
            probabilities
        ))
        sorted_predictions = sorted(
            conditions_with_probs,
            key=lambda x: x[1],
            reverse=True
        )
        
        return prediction[0], sorted_predictions
    
    def process_and_save_predictions(self):
        """
        Load the dataset, handle missing values, make predictions, and save results
        """
        # Load original dataset
        df = pd.read_csv('general_disease_diagnosis.csv')
        
        # Save original dataset
        df.to_csv('original_dataset.csv', index=False)
        
        # Create a copy for predictions
        df_completed = df.copy()
        
        # Handle missing values in features
        for column in self.feature_names:
            if df_completed[column].isnull().any():
                df_completed[column].fillna(df_completed[column].mean(), inplace=True)
        
        # Make predictions for all rows
        predictions = []
        probabilities_list = []
        
        print("\nMaking predictions for all rows...")
        for _, row in df_completed[self.feature_names].iterrows():
            pred, probs = self.predict(*row)
            predictions.append(pred)
            probabilities_list.append(dict(probs))
        
        # Add predictions and probabilities to dataframe
        df_completed['predicted_condition'] = predictions
        
        # Add probability columns for each condition
        for condition in self.label_encoder.classes_:
            df_completed[f'probability_{condition.replace(" ", "_")}'] = [
                probs.get(condition, 0) for probs in probabilities_list
            ]
        
        # Save completed dataset
        df_completed.to_csv('completed_dataset.csv', index=False)
        print("Completed dataset saved as 'completed_dataset.csv'")
        
        return df_completed

# Main execution
if __name__ == "__main__":
    # Create and train model
    print("Initializing and training the model...")
    model = DiseasePredictionModel()
    model.train()
    
    # Process dataset and save predictions
    print("\nProcessing dataset and generating predictions...")
    completed_df = model.process_and_save_predictions()
    
    print("\nProcess completed successfully!")