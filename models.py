"""
Models module for the Movie Summary Analysis application.
Implements machine learning models for genre prediction.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Directory paths
DATA_DIR = "data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class GenrePredictor:
    """Class for predicting movie genres from summaries."""
    
    def __init__(self, model_type='logistic', callback=None):
        """Initialize the GenrePredictor.
        
        Args:
            model_type: Type of model to use ('logistic' or 'random_forest').
            callback: A function to call with progress updates.
        """
        self.model_type = model_type
        self.callback = callback
        self.models = {}
        self.genres_list = []
        
    def train_model(self, X_train, y_train, genres_list):
        """Train a genre prediction model.
        
        Args:
            X_train: Training features.
            y_train: Training labels (multi-label format).
            genres_list: List of genre names.
            
        Returns:
            Trained model dictionary.
        """
        if self.callback:
            self.callback(0)
        
        # Store genres list
        self.genres_list = genres_list
        
        try:
            # Print data dimensions for debugging
            print(f"Training data: X_train={X_train.shape}, y_train={y_train.shape}, genres={len(genres_list)}")
            
            # Check if data is valid
            if X_train.shape[0] == 0 or y_train.shape[0] == 0:
                raise ValueError("Empty training data")
            
            # Create separate classifiers for each genre (binary classification)
            self.models = {}
            
            if self.callback:
                self.callback(10)  # 10% progress
            
            # Train one model per genre
            num_genres = len(genres_list)
            for i, genre in enumerate(genres_list):
                # Extract binary target for this genre
                genre_idx = i
                y_genre = y_train[:, genre_idx]  # Binary target: 0 or 1
                
                # Check if this genre has samples of both classes
                unique_classes = np.unique(y_genre)
                
                # If there's only one class, create a simple classifier that always predicts that class
                if len(unique_classes) < 2:
                    print(f"Warning: Genre '{genre}' has only one class: {unique_classes[0]}. Creating a constant classifier.")
                    
                    # Create a custom classifier that always predicts the only class present
                    class ConstantClassifier:
                        def __init__(self, constant_value):
                            self.constant_value = constant_value
                            
                        def fit(self, X, y):
                            return self
                            
                        def predict(self, X):
                            return np.full(X.shape[0], self.constant_value)
                            
                        def predict_proba(self, X):
                            # For binary classification, return probabilities for both classes
                            # If constant is 0.0, probability for class 0 is 1.0, for class 1 is 0.0
                            # If constant is 1.0, probability for class 0 is 0.0, for class 1 is 1.0
                            probs = np.zeros((X.shape[0], 2))
                            if self.constant_value == 0.0:
                                probs[:, 0] = 1.0  # Probability of class 0
                            else:
                                probs[:, 1] = 1.0  # Probability of class 1
                            return probs
                    
                    # Create and store the constant classifier
                    model = ConstantClassifier(unique_classes[0])
                    model.fit(X_train, y_genre)  # Fit is a no-op but keeps the interface consistent
                    self.models[genre] = model
                else:
                    # Create and train appropriate model for this genre
                    if self.model_type == 'logistic':
                        model = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0)
                    else:  # random_forest
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    # Fit the model
                    model.fit(X_train, y_genre)
                    
                    # Store the trained model
                    self.models[genre] = model
                
                # Update progress
                if self.callback:
                    progress = 10 + int((i / num_genres) * 85)
                    self.callback(progress)
                    
                # Print progress
                if i % 10 == 0 or i == num_genres - 1:
                    print(f"Trained {i+1}/{num_genres} genre classifiers")
            
            if self.callback:
                self.callback(95)  # 95% complete
                
            print(f"Successfully trained {len(self.models)} genre classifiers")
            
            if self.callback:
                self.callback(100)  # 100% complete
                
            return self.models
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            if self.callback:
                self.callback(0)  # Reset progress
            raise
    
    def predict(self, X, threshold=0.3):
        """Predict genres for input features.
        
        Args:
            X: Input features.
            threshold: Probability threshold for classification.
            
        Returns:
            Predicted genres (multi-label format).
        """
        if not self.models:
            raise ValueError("No trained models available. Train the model first.")
        
        # Initialize prediction matrix
        n_samples = X.shape[0]
        n_genres = len(self.genres_list)
        y_pred = np.zeros((n_samples, n_genres))
        
        # Make predictions for each genre
        for i, genre in enumerate(self.genres_list):
            if genre in self.models:
                model = self.models[genre]
                
                if hasattr(model, 'predict_proba'):
                    # Get probability of positive class (index 1)
                    probs = model.predict_proba(X)
                    if probs.shape[1] > 1:  # If we have multiple classes
                        pos_probs = probs[:, 1]
                    else:
                        pos_probs = probs[:, 0]
                        
                    # Apply threshold
                    y_pred[:, i] = (pos_probs >= threshold).astype(int)
                else:
                    # Use default predict method
                    y_pred[:, i] = model.predict(X)
        
        return y_pred
    
    def predict_probabilities(self, X):
        """Predict genre probabilities for input features.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted probabilities for each genre.
        """
        if not self.models:
            raise ValueError("No trained models available. Train the model first.")
        
        # Initialize probability matrix
        n_samples = X.shape[0]
        n_genres = len(self.genres_list)
        probas = np.zeros((n_samples, n_genres))
        
        # Make predictions for each genre
        for i, genre in enumerate(self.genres_list):
            if genre in self.models:
                model = self.models[genre]
                
                try:
                    if hasattr(model, 'predict_proba'):
                        # Get probability of positive class (index 1)
                        probs = model.predict_proba(X)
                        if probs.shape[1] > 1:  # If we have multiple classes
                            probas[:, i] = probs[:, 1]
                        else:
                            probas[:, i] = probs[:, 0]
                    else:
                        # Use binary predictions as probabilities
                        probas[:, i] = model.predict(X)
                except Exception as e:
                    print(f"Error predicting for genre '{genre}': {e}")
        
        return probas
    
    def save_model(self, model_path=None):
        """Save the trained model.
        
        Args:
            model_path: Path to save the model.
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, f"{self.model_type}_model.pkl")
        
        # Check if model is trained
        if not self.models:
            raise ValueError("No trained models to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the models and metadata
        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'genres_list': self.genres_list,
                'model_type': self.model_type
            }, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path=None):
        """Load a trained model.
        
        Args:
            model_path: Path to the model file.
            
        Returns:
            Loaded model dictionary.
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, f"{self.model_type}_model.pkl")
        
        print(f"Loading model from {model_path}")
        
        # Check if file exists and is not empty
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if os.path.getsize(model_path) == 0:
            raise ValueError(f"Model file is empty: {model_path}")
        
        try:
            # Load the model file
            with open(model_path, 'rb') as f:
                try:
                    data = pickle.load(f)
                    
                    # Handle different model formats
                    if isinstance(data, dict):
                        if 'models' in data:
                            self.models = data['models']
                        elif 'estimators' in data:
                            # Convert from previous version
                            self.models = data['estimators']
                        elif 'model' in data and 'genres_list' in data:
                            # Create empty models dict
                            self.models = {}
                            print("Warning: old model format detected. Some features may not work correctly.")
                        
                        # Get genres list
                        if 'genres_list' in data:
                            self.genres_list = data['genres_list']
                        
                        # Get model type if available
                        if 'model_type' in data:
                            self.model_type = data['model_type']
                    else:
                        # If data is not a dictionary, it might be just the models
                        self.models = data
                    
                    return self.models
                except EOFError:
                    print(f"Error: The model file appears to be corrupted or truncated.")
                    # Initialize with empty models
                    self.models = {}
                    self.genres_list = []
                    
                    # Retrain recommendation
                    print("Recommendation: Please retrain the model.")
                    raise ValueError("Model file is corrupted. Please retrain the model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_micro': precision_score(y_test, y_pred, average='micro'),
            'recall_micro': recall_score(y_test, y_pred, average='micro'),
            'f1_micro': f1_score(y_test, y_pred, average='micro'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        }
        
        return metrics
    
    def get_top_genres(self, X, top_n=5):
        """Get the top N predicted genres for input features.
        
        Args:
            X: Input features.
            top_n: Number of top genres to return.
            
        Returns:
            List of (genre, probability) tuples.
        """
        # Get predicted probabilities
        probs = self.predict_probabilities(X)
        
        # Get top genres
        top_indices = np.argsort(probs[0])[::-1][:top_n]
        top_genres = [(self.genres_list[i], probs[0][i]) for i in top_indices]
        
        return top_genres
    def is_model_trained(self):
        """Check if the model is trained.
        
        Returns:
            True if the model is trained, False otherwise.
        """
        model_path = os.path.join(MODELS_DIR, f"{self.model_type}_model.pkl")
        return os.path.exists(model_path)
    
    def plot_confusion_matrix(self, y_true, y_pred, genre_indices=None, figsize=(12, 28)):
        """Plot confusion matrix for selected genres.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            genre_indices: Indices of genres to include in the plot.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure.
        """
        # If genre_indices is not provided, use the top 10 most common genres
        if genre_indices is None:
            genre_counts = y_true.sum(axis=0)
            genre_indices = np.argsort(genre_counts)[::-1][:10]
        
        # Select the genres to include
        selected_genres = [self.genres_list[i] for i in genre_indices]
        y_true_selected = y_true[:, genre_indices]
        y_pred_selected = y_pred[:, genre_indices]
        
        # Create figure with proper spacing
        fig, axes = plt.subplots(len(genre_indices), 1, figsize=figsize)
        
        # If only one genre, wrap the axis in a list
        if len(genre_indices) == 1:
            axes = [axes]
        
        # Plot confusion matrix for each genre
        for i, (genre, ax) in enumerate(zip(selected_genres, axes)):
            cm = confusion_matrix(y_true_selected[:, i], y_pred_selected[:, i])
            
            # Plot the heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            
            # Set title and labels
            ax.set_title(f'Confusion Matrix - {genre}', fontsize=14, pad=20)
            ax.set_xlabel('Predicted', fontsize=12, labelpad=10)
            ax.set_ylabel('True', fontsize=12, labelpad=10)
            
            # Set tick labels
            ax.set_xticklabels(['Negative', 'Positive'], fontsize=11)
            ax.set_yticklabels(['Negative', 'Positive'], fontsize=11)
        
        # Add proper spacing between subplots
        plt.subplots_adjust(hspace=1.0)
        plt.tight_layout(pad=3.0)
        
        return fig
    

