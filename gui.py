"""
GUI module for the Movie Summary Analysis application.
Implements the Tkinter-based user interface.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

from assets.styles import *
from assets.icons import *
from preprocessing import DataPreprocessor
from setup_nltk import setup_nltk as initialize_nltk  # Alias setup_nltk as initialize_nltk
from models import GenrePredictor
from translation_audio import TranslationAudioManager
from utils import (display_matplotlib_figure, create_svg_image, 
                   save_user_summary, load_user_summary, get_all_user_summaries,
                   create_label_with_tooltip)

# Check and create necessary directories
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

class MovieSummaryApp:
    """Main application class for the Movie Summary Analysis GUI."""
    
    def __init__(self, root):
        """Initialize the application.
        
        Args:
            root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Movie Summary Analysis")
        self.root.geometry("900x700")
        self.root.configure(bg=BACKGROUND_COLOR)
        
        # Initialize UI state variables
        self.preprocess_status = None
        self.preprocess_progress = None
        self.model_status = None
        self.model_progress = None
        self.audio_status = None
        self.audio_progress = None
        
        try:
            # Initialize components
            self.preprocessor = DataPreprocessor(callback=self.update_preprocessing_progress)
            self.model = GenrePredictor(callback=self.update_model_progress)
            self.audio_manager = TranslationAudioManager(callback=self.update_audio_progress)
            
            # Setup UI
            self.setup_ui()
        except Exception as e:
            messagebox.showerror("Initialization Error", 
                               f"Error initializing application: {str(e)}\n"
                               "Please ensure all required resources are installed.")
            root.destroy()
            return
        
        # Try to download NLTK resources
        try:
            initialize_nltk()
        except Exception as e:
            messagebox.showwarning("NLTK Download Error", 
                                  f"Error downloading NLTK resources: {e}\n"
                                  "Some functionality may be limited.")
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container frame
        self.main_frame = tk.Frame(self.root, **FRAME_STYLE)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top navigation bar
        self.setup_navigation()
        
        # Content area
        self.content_frame = tk.Frame(self.main_frame, **FRAME_STYLE)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar at the bottom
        self.status_bar = tk.Label(self.main_frame, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initially show the home page
        self.show_home_page()
    
    def setup_navigation(self):
        """Setup the navigation bar."""
        nav_frame = tk.Frame(self.main_frame, bg=SECONDARY_COLOR)
        nav_frame.pack(fill=tk.X, padx=0, pady=0)
        
        # Create navigation buttons
        nav_buttons = [
            ("Home", self.show_home_page, HOME_ICON),
            ("Preprocessing", self.show_preprocessing_page, PREPROCESSING_ICON),
            ("Model", self.show_model_page, MODEL_ICON),
            ("Audio Conversion", self.show_audio_page, AUDIO_ICON),
            ("Genre Prediction", self.show_prediction_page, PREDICTION_ICON)
        ]
        
        # Add buttons to navigation
        for i, (text, command, icon) in enumerate(nav_buttons):
            # Create button frame
            btn_frame = tk.Frame(nav_frame, bg=SECONDARY_COLOR)
            btn_frame.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Create SVG icon
            try:
                icon_image = create_svg_image(icon, fill="white")
                
                # Create button with icon
                icon_label = tk.Label(btn_frame, image=icon_image, bg=SECONDARY_COLOR)
                icon_label.image = icon_image  # Keep a reference
                icon_label.pack(side=tk.LEFT, padx=(5, 0))
                
                # Create button text
                button = tk.Button(btn_frame, text=text, command=command, **NAV_BUTTON_STYLE)
                button.pack(side=tk.LEFT, padx=(0, 5))
            except Exception as e:
                # If icon creation fails, use text-only button
                print(f"Error creating icon: {e}")
                button = tk.Button(btn_frame, text=text, command=command, **NAV_BUTTON_STYLE)
                button.pack(side=tk.LEFT, padx=5)
    
    def clear_content(self):
        """Clear the content frame."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def show_home_page(self):
        """Show the home page."""
        self.clear_content()
        
        # Create welcome message
        welcome_label = tk.Label(self.content_frame, text="Movie Summary Analysis Tool", 
                                **LABEL_STYLE)
        welcome_label.pack(pady=20)
        
        # Create description
        desc_text = """
        This tool allows you to:
        
        • Preprocess movie summaries and metadata
        • Train machine learning models for genre prediction
        • Convert movie summaries to audio in multiple languages
        • Predict genres from new movie summaries
        
        Use the navigation bar above to access different features.
        """
        
        desc_label = tk.Label(self.content_frame, text=desc_text, justify=tk.LEFT, 
                             **LABEL_STYLE)
        desc_label.pack(pady=10)
        
        # Check and show data status
        status_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        status_frame.pack(pady=20, fill=tk.X)
        
        # Check if data is preprocessed
        is_preprocessed = self.preprocessor.is_data_preprocessed()
        preproc_status = "✓ Data is preprocessed" if is_preprocessed else "✗ Data needs preprocessing"
        preproc_color = SUCCESS_COLOR if is_preprocessed else HIGHLIGHT_COLOR
        
        preproc_label = tk.Label(status_frame, text=preproc_status, **LABEL_STYLE)
        preproc_label.config(fg=preproc_color)
        preproc_label.pack(pady=5)
        
        # Check if model is trained
        is_trained = self.model.is_model_trained()
        model_status = "✓ Model is trained" if is_trained else "✗ Model needs training"
        model_color = SUCCESS_COLOR if is_trained else HIGHLIGHT_COLOR
        
        model_label = tk.Label(status_frame, text=model_status, **LABEL_STYLE)
        model_label.config(fg=model_color)
        model_label.pack(pady=5)
        
        # Quick start buttons
        button_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        button_frame.pack(pady=20)
        
        if not is_preprocessed:
            preproc_button = tk.Button(button_frame, text="Start Preprocessing", 
                                      command=self.show_preprocessing_page, **BUTTON_STYLE)
            preproc_button.pack(side=tk.LEFT, padx=10)
        
        if is_preprocessed and not is_trained:
            train_button = tk.Button(button_frame, text="Train Model", 
                                    command=self.show_model_page, **BUTTON_STYLE)
            train_button.pack(side=tk.LEFT, padx=10)
        
        if is_preprocessed:
            predict_button = tk.Button(button_frame, text="Predict Genre", 
                                      command=self.show_prediction_page, **BUTTON_STYLE)
            predict_button.pack(side=tk.LEFT, padx=10)
            
            audio_button = tk.Button(button_frame, text="Convert to Audio", 
                                    command=self.show_audio_page, **BUTTON_STYLE)
            audio_button.pack(side=tk.LEFT, padx=10)
    
    def show_preprocessing_page(self):
        """Show the preprocessing page."""
        self.clear_content()
        
        # Create preprocessing title
        title_label = tk.Label(self.content_frame, text="Data Preprocessing", 
                              **LABEL_STYLE)
        title_label.pack(pady=10)
        
        # Check if data is already preprocessed
        is_preprocessed = self.preprocessor.is_data_preprocessed()
        
        if is_preprocessed:
            status_label = tk.Label(self.content_frame, 
                                  text="Data has already been preprocessed.", 
                                  **LABEL_STYLE)
            status_label.config(fg=SUCCESS_COLOR)
            status_label.pack(pady=10)
            
            # Option to reprocess data
            reprocess_button = tk.Button(self.content_frame, text="Reprocess Data", 
                                       command=self.start_preprocessing, **BUTTON_STYLE)
            reprocess_button.pack(pady=10)
        else:
            # Description of preprocessing
            desc_text = """
            Preprocessing includes:
            
            • Cleaning movie summaries (removing special characters, stopwords)
            • Tokenizing and lemmatizing text
            • Extracting genre information from metadata
            • Creating TF-IDF vectorizer for feature extraction
            
            This process may take some time.
            """
            
            desc_label = tk.Label(self.content_frame, text=desc_text, justify=tk.LEFT, 
                                 **LABEL_STYLE)
            desc_label.pack(pady=10)
            
            # Progress bar for preprocessing
            self.preprocess_progress = ttk.Progressbar(self.content_frame, orient=tk.HORIZONTAL, 
                                                     length=400, mode='determinate')
            self.preprocess_progress.pack(pady=10, fill=tk.X, padx=50)
            
            # Progress status label
            self.preprocess_status = tk.Label(self.content_frame, text="Ready to start preprocessing", 
                                           **LABEL_STYLE)
            self.preprocess_status.pack(pady=5)
            
            # Start preprocessing button
            start_button = tk.Button(self.content_frame, text="Start Preprocessing", 
                                   command=self.start_preprocessing, **BUTTON_STYLE)
            start_button.pack(pady=10)
    
    def start_preprocessing(self):
        """Start the preprocessing process in a separate thread."""
        # Check if we have the necessary UI elements
        if not hasattr(self, 'preprocess_status') or self.preprocess_status is None:
            # If not, show the preprocessing page first
            self.show_preprocessing_page()
            # Wait a bit for the UI to update before continuing
            self.root.update()
            
        # Disable UI elements during preprocessing
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.DISABLED)
        
        # Update status
        if hasattr(self, 'preprocess_status') and self.preprocess_status is not None:
            self.preprocess_status.config(text="Preprocessing started...")
        if hasattr(self, 'preprocess_progress') and self.preprocess_progress is not None:
            self.preprocess_progress['value'] = 0
        
        # Start preprocessing in a separate thread
        threading.Thread(target=self.run_preprocessing, daemon=True).start()
    
    def run_preprocessing(self):
        """Run the preprocessing process."""
        try:
            # Run preprocessing
            self.preprocessor.preprocess_data()
            
            # Update UI
            self.root.after(0, self.preprocessing_complete)
        except Exception as e:
            # Handle errors
            error_message = f"Error during preprocessing: {str(e)}"
            self.root.after(0, lambda: self.preprocessing_error(error_message))
    
    def update_preprocessing_progress(self, progress):
        """Update the preprocessing progress bar.
        
        Args:
            progress: Progress value (0-100).
        """
        # Only update if the UI elements exist
        if hasattr(self, 'preprocess_progress') and self.preprocess_progress is not None:
            self.root.after(0, lambda: self.preprocess_progress.config(value=progress))
        
        if hasattr(self, 'preprocess_status') and self.preprocess_status is not None:
            self.root.after(0, lambda: self.preprocess_status.config(
                text=f"Preprocessing: {progress}% complete"))
    
    def preprocessing_complete(self):
        """Handle preprocessing completion."""
        # Update status
        self.preprocess_status.config(text="Preprocessing complete!", fg=SUCCESS_COLOR)
        self.preprocess_progress['value'] = 100
        
        # Enable UI elements
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.NORMAL)
        
        # Add button to continue to model training
        continue_button = tk.Button(self.content_frame, text="Continue to Model Training", 
                                  command=self.show_model_page, **BUTTON_STYLE)
        continue_button.pack(pady=10)
        
        # Update status bar
        self.status_bar.config(text="Preprocessing complete")
    
    def preprocessing_error(self, error_message):
        """Handle preprocessing errors.
        
        Args:
            error_message: Error message to display.
        """
        # Update status
        self.preprocess_status.config(text=error_message, fg=HIGHLIGHT_COLOR)
        
        # Enable UI elements
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.NORMAL)
        
        # Update status bar
        self.status_bar.config(text="Preprocessing failed")
        
        # Show error message
        messagebox.showerror("Preprocessing Error", error_message)
    
    def show_model_page(self):
        """Show the model page."""
        self.clear_content()
        
        # Create model title
        title_label = tk.Label(self.content_frame, text="Genre Prediction Model", 
                              **LABEL_STYLE)
        title_label.pack(pady=10)
        
        # Check if data is preprocessed
        if not self.preprocessor.is_data_preprocessed():
            error_label = tk.Label(self.content_frame, 
                                 text="Data needs to be preprocessed first!", 
                                 **LABEL_STYLE)
            error_label.config(fg=HIGHLIGHT_COLOR)
            error_label.pack(pady=10)
            
            preproc_button = tk.Button(self.content_frame, text="Go to Preprocessing", 
                                     command=self.show_preprocessing_page, **BUTTON_STYLE)
            preproc_button.pack(pady=10)
            return
        
        # Check if model is already trained
        is_trained = self.model.is_model_trained()
        
        # Model selection frame
        model_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        model_frame.pack(pady=10, fill=tk.X)
        
        model_label = tk.Label(model_frame, text="Select Model Type:", **LABEL_STYLE)
        model_label.pack(side=tk.LEFT, padx=10)
        
        # Model type variable
        self.model_type_var = tk.StringVar(value="logistic")
        
        # Model type radiobuttons
        logistic_radio = tk.Radiobutton(model_frame, text="Logistic Regression", 
                                      variable=self.model_type_var, value="logistic", 
                                      **LABEL_STYLE)
        logistic_radio.pack(side=tk.LEFT, padx=10)
        
        rf_radio = tk.Radiobutton(model_frame, text="Random Forest", 
                                variable=self.model_type_var, value="random_forest", 
                                **LABEL_STYLE)
        rf_radio.pack(side=tk.LEFT, padx=10)
        
        # Add info tooltip about model selection
        model_info = create_label_with_tooltip(
            model_frame, 
            text="ℹ️", 
            tooltip_text="Logistic Regression: Fast, interpretable, works well with text.\n"
                        "Random Forest: More complex, better accuracy, slower to train.",
            **LABEL_STYLE
        )
        model_info.pack(side=tk.LEFT, padx=5)
        
        # Model explanation frame
        explanation_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        explanation_frame.pack(pady=10, fill=tk.X)
        
        explanation_text = """
        Model Information:
        
        • Logistic Regression: A linear model that works well for text classification tasks. 
          It's fast to train and provides interpretable results with good performance.
          
        • Random Forest: An ensemble of decision trees that can capture more complex patterns.
          It's more robust but slower to train and less interpretable.
          
        The training process includes:
        • Feature extraction using TF-IDF
        • Multi-label classification handling
        • Training with 80% data, testing with 20%
        • Evaluation using precision, recall, and F1-score
        """
        
        explanation_label = tk.Label(explanation_frame, text=explanation_text, 
                                   justify=tk.LEFT, **LABEL_STYLE)
        explanation_label.pack(pady=10, fill=tk.X)
        
        # Progress bar for model training
        self.model_progress = ttk.Progressbar(self.content_frame, orient=tk.HORIZONTAL, 
                                            length=400, mode='determinate')
        self.model_progress.pack(pady=10, fill=tk.X, padx=50)
        
        # Progress status label
        self.model_status = tk.Label(self.content_frame, text="Ready to train model", 
                                   **LABEL_STYLE)
        self.model_status.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        button_frame.pack(pady=10)
        
        # Status message
        if is_trained:
            status_label = tk.Label(button_frame, text="Model has already been trained. ", 
                                  **LABEL_STYLE)
            status_label.config(fg=SUCCESS_COLOR)
            
            # Add retrain button
            train_button = tk.Button(button_frame, text="Retrain Model", 
                                   command=self.start_model_training, **BUTTON_STYLE)
            train_button.pack(side=tk.LEFT, padx=5)
            
            # Add evaluate button
            evaluate_button = tk.Button(button_frame, text="Evaluate Model", 
                                      command=self.evaluate_model, **BUTTON_STYLE)
            evaluate_button.pack(side=tk.LEFT, padx=5)
        else:
            # Start training button
            train_button = tk.Button(button_frame, text="Train Model", 
                                   command=self.start_model_training, **BUTTON_STYLE)
            train_button.pack(side=tk.LEFT, padx=5)
    
    def start_model_training(self):
        """Start the model training process in a separate thread."""
        # Check if we have the necessary UI elements
        if not hasattr(self, 'model_status') or self.model_status is None:
            # If not, show the model page first
            self.show_model_page()
            # Wait for the UI to update
            self.root.update()
            
        # Update model type
        model_type = self.model_type_var.get()
        self.model = GenrePredictor(model_type=model_type, callback=self.update_model_progress)
        
        # Disable UI elements during training
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.DISABLED)
        
        # Update status
        if hasattr(self, 'model_status') and self.model_status is not None:
            self.model_status.config(text=f"Training {model_type} model...")
        if hasattr(self, 'model_progress') and self.model_progress is not None:
            self.model_progress['value'] = 0
        
        # Start training in a separate thread
        threading.Thread(target=self.run_model_training, daemon=True).start()
    
    def run_model_training(self):
        """Run the model training process."""
        try:
            # Prepare training data
            try:
                X_train, X_test, y_train, y_test, genres_list, vectorizer = self.preprocessor.prepare_training_data()
                print(f"Training data prepared: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
                print(f"Number of genres: {len(genres_list)}")
            except Exception as e:
                raise ValueError(f"Error preparing training data: {str(e)}")
            
            # Train model
            try:
                self.model.train_model(X_train, y_train, genres_list)
                print("Model training completed successfully")
            except Exception as e:
                raise ValueError(f"Error during model training: {str(e)}")
            
            # Save model
            try:
                self.model.save_model()
                print("Model saved successfully")
            except Exception as e:
                print(f"Warning: Error saving model: {str(e)}")
            
            # Evaluate model
            try:
                self.metrics = self.model.evaluate_model(X_test, y_test)
                print("Model evaluation completed successfully")
            except Exception as e:
                print(f"Warning: Error during model evaluation: {str(e)}")
                self.metrics = {
                    'accuracy': 0,
                    'precision_micro': 0,
                    'recall_micro': 0,
                    'f1_micro': 0,
                    'precision_macro': 0,
                    'recall_macro': 0,
                    'f1_macro': 0
                }
            
            # Make predictions for confusion matrix
            try:
                self.y_test = y_test
                self.y_pred = self.model.predict(X_test)
                print("Predictions for evaluation completed successfully")
            except Exception as e:
                print(f"Warning: Error making predictions: {str(e)}")
                self.y_test = np.zeros((100, len(genres_list)))
                self.y_pred = np.zeros((100, len(genres_list)))
            
            # Update UI
            self.root.after(0, self.model_training_complete)
        except Exception as e:
            # Handle errors
            error_message = f"Error during model training: {str(e)}"
            print(f"ERROR: {error_message}")
            self.root.after(0, lambda: self.model_training_error(error_message))
    
    def update_model_progress(self, progress):
        """Update the model training progress bar.
        
        Args:
            progress: Progress value (0-100).
        """
        # Only update if the UI elements exist
        if hasattr(self, 'model_progress') and self.model_progress is not None:
            self.root.after(0, lambda: self.model_progress.config(value=progress))
        
        if hasattr(self, 'model_status') and self.model_status is not None:
            self.root.after(0, lambda: self.model_status.config(
                text=f"Training: {progress}% complete"))
    
    def model_training_complete(self):
        """Handle model training completion."""
        # Update status
        self.model_status.config(text="Model training complete!", fg=SUCCESS_COLOR)
        self.model_progress['value'] = 100
        
        # Enable UI elements
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.NORMAL)
        
        # Show evaluation metrics
        metrics_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        metrics_frame.pack(pady=10, fill=tk.X)
        
        metrics_title = tk.Label(metrics_frame, text="Model Evaluation Metrics", 
                               **LABEL_STYLE)
        metrics_title.pack(pady=5)
        
        # Format metrics
        metrics_text = f"""
        Accuracy: {self.metrics['accuracy']:.4f}
        
        Micro-average metrics (treats all classes equally):
        Precision: {self.metrics['precision_micro']:.4f}
        Recall: {self.metrics['recall_micro']:.4f}
        F1-Score: {self.metrics['f1_micro']:.4f}
        
        Macro-average metrics (average over all classes):
        Precision: {self.metrics['precision_macro']:.4f}
        Recall: {self.metrics['recall_macro']:.4f}
        F1-Score: {self.metrics['f1_macro']:.4f}
        """
        
        metrics_label = tk.Label(metrics_frame, text=metrics_text, justify=tk.LEFT, 
                               **LABEL_STYLE)
        metrics_label.pack(pady=5)
        
        # Create evaluation button
        evaluate_button = tk.Button(self.content_frame, text="View Confusion Matrix", 
                                  command=self.evaluate_model, **BUTTON_STYLE)
        evaluate_button.pack(pady=10)
        
        # Continue to prediction button
        continue_button = tk.Button(self.content_frame, text="Continue to Genre Prediction", 
                                  command=self.show_prediction_page, **BUTTON_STYLE)
        continue_button.pack(pady=10)
        
        # Update status bar
        self.status_bar.config(text="Model training complete")
    
    def model_training_error(self, error_message):
        """Handle model training errors.
        
        Args:
            error_message: Error message to display.
        """
        # Update status
        self.model_status.config(text=error_message, fg=HIGHLIGHT_COLOR)
        
        # Enable UI elements
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.NORMAL)
        
        # Update status bar
        self.status_bar.config(text="Model training failed")
        
        # Show error message
        messagebox.showerror("Training Error", error_message)
    
    def evaluate_model(self):
        """Show model evaluation results."""
        # Create popup window
        eval_window = tk.Toplevel(self.root)
        eval_window.title("Model Evaluation")
        eval_window.geometry("800x600")
        eval_window.configure(bg=BACKGROUND_COLOR)
        
        # Title
        title_label = tk.Label(eval_window, text="Confusion Matrices for Top Genres", 
                             font=("Helvetica", 16, "bold"), bg=BACKGROUND_COLOR)
        title_label.pack(pady=10)
        
        # If model is trained but evaluation data isn't stored, train again
        if not hasattr(self, 'y_test') or not hasattr(self, 'y_pred'):
            try:
                # Load model
                self.model.load_model()
                
                # Prepare data
                X_train, X_test, y_train, y_test, genres_list, vectorizer = self.preprocessor.prepare_training_data()
                
                # Make predictions
                self.y_test = y_test
                self.y_pred = self.model.predict(X_test)
                
                # Calculate metrics
                self.metrics = self.model.evaluate_model(X_test, y_test)
            except Exception as e:
                error_message = f"Error loading model for evaluation: {str(e)}"
                messagebox.showerror("Evaluation Error", error_message)
                eval_window.destroy()
                return
        
        # Create a figure for the confusion matrix
        fig = self.model.plot_confusion_matrix(self.y_test, self.y_pred)
        
        # Create a frame for the plot
        plot_frame = tk.Frame(eval_window, bg=BACKGROUND_COLOR)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Display the figure
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Close button
        close_button = tk.Button(eval_window, text="Close", 
                               command=eval_window.destroy, **BUTTON_STYLE)
        close_button.pack(pady=10)
    
    def show_audio_page(self):
        """Show the audio conversion page."""
        self.clear_content()
        
        # Create audio title
        title_label = tk.Label(self.content_frame, text="Text-to-Speech Conversion", 
                              **LABEL_STYLE)
        title_label.pack(pady=10)
        
        # Create two sections: left for input, right for output
        main_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left section - Input
        input_frame = tk.Frame(main_frame, **FRAME_STYLE)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        input_label = tk.Label(input_frame, text="Enter Movie Summary:", **LABEL_STYLE)
        input_label.pack(pady=(0, 5), anchor=tk.W)
        
        self.summary_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, 
                                                   height=15, **TEXT_STYLE)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Language selection
        language_frame = tk.Frame(input_frame, **FRAME_STYLE)
        language_frame.pack(fill=tk.X, pady=10)
        
        language_label = tk.Label(language_frame, text="Select Language:", **LABEL_STYLE)
        language_label.pack(side=tk.LEFT, padx=5)
        
        self.language_var = tk.StringVar(value="English")
        languages = ["English", "Arabic", "Urdu", "Korean"]
        
        language_menu = tk.OptionMenu(language_frame, self.language_var, *languages)
        language_menu.config(bg=PRIMARY_COLOR, fg="white", activebackground=SECONDARY_COLOR)
        language_menu.pack(side=tk.LEFT, padx=5)
        
        # Convert button
        convert_button = tk.Button(input_frame, text="Convert to Speech", 
                                 command=self.convert_to_speech, **BUTTON_STYLE)
        convert_button.pack(pady=10)
        
        # Right section - Output
        output_frame = tk.Frame(main_frame, **FRAME_STYLE)
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        output_label = tk.Label(output_frame, text="Audio Controls:", **LABEL_STYLE)
        output_label.pack(pady=(0, 5), anchor=tk.W)
        
        # Progress bar for audio conversion
        self.audio_progress = ttk.Progressbar(output_frame, orient=tk.HORIZONTAL, 
                                           length=300, mode='determinate')
        self.audio_progress.pack(pady=10, fill=tk.X)
        
        # Status label
        self.audio_status = tk.Label(output_frame, text="Ready to convert", **LABEL_STYLE)
        self.audio_status.pack(pady=5)
        
        # Audio controls
        controls_frame = tk.Frame(output_frame, **FRAME_STYLE)
        controls_frame.pack(pady=10)
        
        # Play button
        self.play_button = tk.Button(controls_frame, text="Play", 
                                  command=self.play_audio, state=tk.DISABLED, **BUTTON_STYLE)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_button = tk.Button(controls_frame, text="Stop", 
                                  command=self.stop_audio, state=tk.DISABLED, **BUTTON_STYLE)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.save_button = tk.Button(controls_frame, text="Save", 
                                  command=self.save_audio, state=tk.DISABLED, **BUTTON_STYLE)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Batch processing section
        batch_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        batch_frame.pack(fill=tk.X, pady=10)
        
        batch_label = tk.Label(batch_frame, text="Batch Processing", 
                             **LABEL_STYLE)
        batch_label.pack(pady=5)
        
        batch_desc = tk.Label(batch_frame, 
                           text="Convert multiple summaries from the dataset to audio files.", 
                           **LABEL_STYLE)
        batch_desc.pack(pady=5)
        
        batch_button = tk.Button(batch_frame, text="Process Batch of Summaries", 
                               command=self.show_batch_processing, **BUTTON_STYLE)
        batch_button.pack(pady=5)
    
    def convert_to_speech(self):
        """Convert text to speech."""
        # Get summary text
        summary = self.summary_text.get("1.0", tk.END).strip()
        
        if not summary:
            messagebox.showwarning("Empty Summary", "Please enter a movie summary.")
            return
        
        # Get selected language
        language = self.language_var.get()
        
        # Disable controls during conversion
        if hasattr(self, 'play_button') and self.play_button is not None:
            self.play_button.config(state=tk.DISABLED)
        if hasattr(self, 'stop_button') and self.stop_button is not None:
            self.stop_button.config(state=tk.DISABLED)
        if hasattr(self, 'save_button') and self.save_button is not None:
            self.save_button.config(state=tk.DISABLED)
        
        # Reset progress bar
        if hasattr(self, 'audio_progress') and self.audio_progress is not None:
            self.audio_progress['value'] = 0
        if hasattr(self, 'audio_status') and self.audio_status is not None:
            self.audio_status.config(text=f"Converting to {language}...")
        
        # Start conversion in a separate thread
        threading.Thread(target=self.run_conversion, args=(summary, language), daemon=True).start()
    
    def run_conversion(self, summary, language):
        """Run the text-to-speech conversion.
        
        Args:
            summary: The text to convert.
            language: The target language.
        """
        try:
            # Convert to speech
            self.audio_path = self.audio_manager.translate_and_speak(summary, language)
            
            # Update UI
            self.root.after(0, self.conversion_complete)
        except Exception as e:
            # Handle errors
            error_message = f"Error during conversion: {str(e)}"
            self.root.after(0, lambda: self.conversion_error(error_message))
    
    def update_audio_progress(self, progress, status_text=None):
        """Update the audio conversion progress bar.
        
        Args:
            progress: Progress value (0-100).
            status_text: Optional status text.
        """
        # Only update if the UI elements exist
        if hasattr(self, 'audio_progress') and self.audio_progress is not None:
            self.root.after(0, lambda: self.audio_progress.config(value=progress))
        
        if hasattr(self, 'audio_status') and self.audio_status is not None:
            if status_text:
                self.root.after(0, lambda: self.audio_status.config(text=status_text))
            else:
                self.root.after(0, lambda: self.audio_status.config(
                    text=f"Converting: {progress}% complete"))
    
    def conversion_complete(self):
        """Handle conversion completion."""
        # Update status
        self.audio_status.config(text="Conversion complete!", fg=SUCCESS_COLOR)
        self.audio_progress['value'] = 100
        
        # Enable controls
        self.play_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        
        # Update status bar
        self.status_bar.config(text="Audio conversion complete")
    
    def conversion_error(self, error_message):
        """Handle conversion errors.
        
        Args:
            error_message: Error message to display.
        """
        # Update status
        self.audio_status.config(text=error_message, fg=HIGHLIGHT_COLOR)
        
        # Update status bar
        self.status_bar.config(text="Audio conversion failed")
        
        # Show error message
        messagebox.showerror("Conversion Error", error_message)
    
    def play_audio(self):
        """Play the converted audio."""
        if hasattr(self, 'audio_path') and self.audio_path:
            # Play audio
            self.audio_manager.play_audio(self.audio_path)
            
            # Update status
            self.audio_status.config(text="Playing audio...")
    
    def stop_audio(self):
        """Stop audio playback."""
        self.audio_manager.stop_audio()
        
        # Update status
        self.audio_status.config(text="Audio stopped")
    
    def save_audio(self):
        """Save the audio file."""
        if hasattr(self, 'audio_path') and self.audio_path:
            # Get default filename
            default_name = os.path.basename(self.audio_path)
            
            # Show save dialog
            from tkinter import filedialog
            save_path = filedialog.asksaveasfilename(
                defaultextension=".mp3",
                filetypes=[("MP3 Files", "*.mp3")],
                initialfile=default_name
            )
            
            if save_path:
                try:
                    # Copy file
                    import shutil
                    shutil.copy2(self.audio_path, save_path)
                    
                    # Update status
                    self.audio_status.config(text=f"Audio saved to {save_path}", fg=SUCCESS_COLOR)
                except Exception as e:
                    # Handle errors
                    error_message = f"Error saving audio: {str(e)}"
                    self.audio_status.config(text=error_message, fg=HIGHLIGHT_COLOR)
                    messagebox.showerror("Save Error", error_message)
    
    def show_batch_processing(self):
        """Show batch processing dialog."""
        # Create popup window
        batch_window = tk.Toplevel(self.root)
        batch_window.title("Batch Audio Processing")
        batch_window.geometry("600x500")
        batch_window.configure(bg=BACKGROUND_COLOR)
        
        # Title
        title_label = tk.Label(batch_window, text="Batch Audio Processing", 
                             font=("Helvetica", 16, "bold"), bg=BACKGROUND_COLOR)
        title_label.pack(pady=10)
        
        # Check if data is preprocessed
        if not self.preprocessor.is_data_preprocessed():
            error_label = tk.Label(batch_window, 
                                 text="Data needs to be preprocessed first!", 
                                 **LABEL_STYLE)
            error_label.config(fg=HIGHLIGHT_COLOR)
            error_label.pack(pady=10)
            
            close_button = tk.Button(batch_window, text="Close", 
                                   command=batch_window.destroy, **BUTTON_STYLE)
            close_button.pack(pady=10)
            return
        
        # Description
        desc_label = tk.Label(batch_window, 
                           text="Select the number of summaries and target languages:", 
                           bg=BACKGROUND_COLOR)
        desc_label.pack(pady=10)
        
        # Options frame
        options_frame = tk.Frame(batch_window, bg=BACKGROUND_COLOR)
        options_frame.pack(pady=10, fill=tk.X)
        
        # Number of summaries
        count_frame = tk.Frame(options_frame, bg=BACKGROUND_COLOR)
        count_frame.pack(pady=5, fill=tk.X)
        
        count_label = tk.Label(count_frame, text="Number of summaries:", bg=BACKGROUND_COLOR)
        count_label.pack(side=tk.LEFT, padx=10)
        
        self.batch_count_var = tk.StringVar(value="10")
        count_entry = tk.Entry(count_frame, textvariable=self.batch_count_var, width=5)
        count_entry.pack(side=tk.LEFT, padx=5)
        
        # Language selection
        language_frame = tk.Frame(options_frame, bg=BACKGROUND_COLOR)
        language_frame.pack(pady=5, fill=tk.X)
        
        language_label = tk.Label(language_frame, text="Languages:", bg=BACKGROUND_COLOR)
        language_label.pack(side=tk.LEFT, padx=10)
        
        # Language checkboxes
        self.lang_vars = {}
        languages = ["English", "Arabic", "Urdu", "Korean"]
        
        for lang in languages:
            var = tk.BooleanVar(value=True)
            self.lang_vars[lang] = var
            
            cb = tk.Checkbutton(language_frame, text=lang, variable=var, bg=BACKGROUND_COLOR)
            cb.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.batch_progress = ttk.Progressbar(batch_window, orient=tk.HORIZONTAL, 
                                           length=500, mode='determinate')
        self.batch_progress.pack(pady=10, fill=tk.X, padx=20)
        
        # Status label
        self.batch_status = tk.Label(batch_window, text="Ready to process", bg=BACKGROUND_COLOR)
        self.batch_status.pack(pady=5)
        
        # Results frame
        results_frame = tk.Frame(batch_window, bg=BACKGROUND_COLOR)
        results_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)
        
        results_label = tk.Label(results_frame, text="Results:", bg=BACKGROUND_COLOR)
        results_label.pack(anchor=tk.W)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                   height=10, **TEXT_STYLE)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(batch_window, bg=BACKGROUND_COLOR)
        button_frame.pack(pady=10)
        
        # Start button
        start_button = tk.Button(button_frame, text="Start Processing", 
                               command=lambda: self.start_batch_processing(batch_window), 
                               **BUTTON_STYLE)
        start_button.pack(side=tk.LEFT, padx=10)
        
        # Close button
        close_button = tk.Button(button_frame, text="Close", 
                               command=batch_window.destroy, **BUTTON_STYLE)
        close_button.pack(side=tk.LEFT, padx=10)
    
    def start_batch_processing(self, window):
        """Start batch processing.
        
        Args:
            window: The batch processing window.
        """
        # Get options
        try:
            count = int(self.batch_count_var.get())
            if count <= 0:
                raise ValueError("Count must be positive")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid number of summaries: {str(e)}")
            return
        
        # Get selected languages
        selected_languages = [lang for lang, var in self.lang_vars.items() if var.get()]
        
        if not selected_languages:
            messagebox.showwarning("No Languages", "Please select at least one language.")
            return
        
        # Disable controls
        for widget in window.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.DISABLED)
        
        # Reset progress and status
        self.batch_progress['value'] = 0
        self.batch_status.config(text="Loading data...")
        self.results_text.delete("1.0", tk.END)
        
        # Start processing in a separate thread
        threading.Thread(target=self.run_batch_processing, 
                       args=(window, count, selected_languages), daemon=True).start()
    
    def run_batch_processing(self, window, count, languages):
        """Run batch processing.
        
        Args:
            window: The batch processing window.
            count: Number of summaries to process.
            languages: List of target languages.
        """
        try:
            # Load preprocessed data
            df = self.preprocessor.load_preprocessed_data()
            
            # Select random samples
            samples = df.sample(min(count, len(df)))
            
            # Process each sample
            results = []
            for i, (_, row) in enumerate(samples.iterrows()):
                # Update progress
                progress = int((i / len(samples)) * 100)
                self.root.after(0, lambda p=progress: self.batch_progress.config(value=p))
                self.root.after(0, lambda i=i, total=len(samples): 
                             self.batch_status.config(text=f"Processing {i+1}/{total}..."))
                
                # Get summary and movie ID
                summary = row['original_summary']
                movie_id = row['movie_id']
                
                # Process the summary
                try:
                    # Translate and convert to speech for each language
                    audio_files = {}
                    for language in languages:
                        lang_code = self.audio_manager.languages.get(language, 'en')
                        
                        # Update status
                        self.root.after(0, lambda lang=language: 
                                     self.batch_status.config(
                                         text=f"Processing {i+1}/{len(samples)} - {lang}..."))
                        
                        # Translate
                        translated = self.audio_manager.translate_text(summary[:500], lang_code)
                        
                        if translated:
                            # Generate filename
                            filename = f"batch_movie_{movie_id}_{lang_code}.mp3"
                            
                            # Convert to speech
                            audio_path = self.audio_manager.text_to_speech(
                                translated, lang_code, filename)
                            
                            if audio_path:
                                audio_files[language] = audio_path
                    
                    # Add to results
                    results.append({
                        'movie_id': movie_id,
                        'audio_files': audio_files
                    })
                    
                    # Update results text
                    result_text = f"Movie {movie_id}: Processed in {', '.join(audio_files.keys())}\n"
                    self.root.after(0, lambda txt=result_text: 
                                 self.results_text.insert(tk.END, txt))
                except Exception as e:
                    # Log error
                    error_text = f"Error processing movie {movie_id}: {str(e)}\n"
                    self.root.after(0, lambda txt=error_text: 
                                 self.results_text.insert(tk.END, txt))
            
            # Update UI
            self.root.after(0, lambda: self.batch_processing_complete(window, results))
        except Exception as e:
            # Handle errors
            error_message = f"Error during batch processing: {str(e)}"
            self.root.after(0, lambda: self.batch_processing_error(window, error_message))
    
    def batch_processing_complete(self, window, results):
        """Handle batch processing completion.
        
        Args:
            window: The batch processing window.
            results: Processing results.
        """
        # Update status
        self.batch_status.config(text=f"Processing complete! {len(results)} summaries processed.", 
                              fg=SUCCESS_COLOR)
        self.batch_progress['value'] = 100
        
        # Enable controls
        for widget in window.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.NORMAL)
        
        # Add summary to results
        summary_text = f"\nSummary: {len(results)} movies processed successfully.\n"
        self.results_text.insert(tk.END, summary_text)
        
        # Update status bar
        self.status_bar.config(text=f"Batch processing complete: {len(results)} summaries")
    
    def batch_processing_error(self, window, error_message):
        """Handle batch processing errors.
        
        Args:
            window: The batch processing window.
            error_message: Error message to display.
        """
        # Update status
        self.batch_status.config(text=error_message, fg=HIGHLIGHT_COLOR)
        
        # Enable controls
        for widget in window.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(state=tk.NORMAL)
        
        # Add error to results
        self.results_text.insert(tk.END, f"\nError: {error_message}\n")
        
        # Update status bar
        self.status_bar.config(text="Batch processing failed")
        
        # Show error message
        messagebox.showerror("Batch Processing Error", error_message)
    
    def show_prediction_page(self):
        """Show the genre prediction page."""
        self.clear_content()
        
        # Create prediction title
        title_label = tk.Label(self.content_frame, text="Movie Genre Prediction", 
                              **LABEL_STYLE)
        title_label.pack(pady=10)
        
        # Check if data is preprocessed and model is trained
        if not self.preprocessor.is_data_preprocessed():
            error_label = tk.Label(self.content_frame, 
                                 text="Data needs to be preprocessed first!", 
                                 **LABEL_STYLE)
            error_label.config(fg=HIGHLIGHT_COLOR)
            error_label.pack(pady=10)
            
            preproc_button = tk.Button(self.content_frame, text="Go to Preprocessing", 
                                     command=self.show_preprocessing_page, **BUTTON_STYLE)
            preproc_button.pack(pady=10)
            return
        
        if not self.model.is_model_trained():
            error_label = tk.Label(self.content_frame, 
                                 text="Model needs to be trained first!", 
                                 **LABEL_STYLE)
            error_label.config(fg=HIGHLIGHT_COLOR)
            error_label.pack(pady=10)
            
            train_button = tk.Button(self.content_frame, text="Go to Model Training", 
                                   command=self.show_model_page, **BUTTON_STYLE)
            train_button.pack(pady=10)
            return
        
        # Create two sections: left for input, right for output
        main_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left section - Input
        input_frame = tk.Frame(main_frame, **FRAME_STYLE)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        input_label = tk.Label(input_frame, text="Enter Movie Summary:", **LABEL_STYLE)
        input_label.pack(pady=(0, 5), anchor=tk.W)
        
        self.prediction_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, 
                                                      height=15, **TEXT_STYLE)
        self.prediction_text.pack(fill=tk.BOTH, expand=True)
        
        # Predict button
        predict_button = tk.Button(input_frame, text="Predict Genre", 
                                 command=self.predict_genre, **BUTTON_STYLE)
        predict_button.pack(pady=10)
        
        # Example button
        example_button = tk.Button(input_frame, text="Load Example", 
                                 command=self.load_prediction_example, **BUTTON_STYLE)
        example_button.pack(pady=10)
        
        # Right section - Output
        output_frame = tk.Frame(main_frame, **FRAME_STYLE)
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        output_label = tk.Label(output_frame, text="Predicted Genres:", **LABEL_STYLE)
        output_label.pack(pady=(0, 5), anchor=tk.W)
        
        # Results frame
        self.results_frame = tk.Frame(output_frame, **FRAME_STYLE)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        initial_label = tk.Label(self.results_frame, 
                               text="Enter a movie summary and click 'Predict Genre'", 
                               **LABEL_STYLE)
        initial_label.pack(pady=20)
        
        # Additional options
        options_frame = tk.Frame(self.content_frame, **FRAME_STYLE)
        options_frame.pack(fill=tk.X, pady=10)
        
        # Convert to audio option
        audio_button = tk.Button(options_frame, text="Convert Summary to Audio", 
                               command=self.prediction_to_audio, state=tk.DISABLED, **BUTTON_STYLE)
        audio_button.pack(side=tk.LEFT, padx=10)
        self.audio_button = audio_button
        
        # Save prediction option
        save_button = tk.Button(options_frame, text="Save Prediction", 
                              command=self.save_prediction, state=tk.DISABLED, **BUTTON_STYLE)
        save_button.pack(side=tk.LEFT, padx=10)
        self.save_button = save_button
    
    def predict_genre(self):
        """Predict genre for the entered summary."""
        # Get summary text
        summary = self.prediction_text.get("1.0", tk.END).strip()
        
        if not summary:
            messagebox.showwarning("Empty Summary", "Please enter a movie summary.")
            return
        
        # Update status
        self.status_bar.config(text="Predicting genre...")
        
        try:
            # Check if model exists
            if not os.path.exists(os.path.join("data", "models", f"{self.model.model_type}_model.pkl")):
                messagebox.showerror("Model Not Found", 
                                   "The model file doesn't exist. Please train the model first.")
                self.status_bar.config(text="Prediction failed - model not found")
                return
            
            # Load model if not already loaded
            if not hasattr(self.model, 'models') or not self.model.models:
                try:
                    self.model.load_model()
                except (FileNotFoundError, ValueError) as e:
                    messagebox.showerror("Model Error", 
                                       f"{str(e)}\n\nPlease go to the Model page and train the model.")
                    self.status_bar.config(text="Prediction failed - model error")
                    return
            
            # Preprocess summary
            summary_vector = self.preprocessor.preprocess_new_summary(summary)
            
            # Predict genre
            self.prediction_results = self.model.get_top_genres(summary_vector)
            
            # Display results
            self.display_prediction_results()
            
            # Enable additional options
            self.audio_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            
            # Update status
            self.status_bar.config(text="Genre prediction complete")
        except Exception as e:
            # Handle errors and print detailed exception for debugging
            import traceback
            error_details = traceback.format_exc()
            print(f"Prediction error details: {error_details}")
            
            error_message = f"Error during prediction: {str(e)}"
            messagebox.showerror("Prediction Error", error_message)
            self.status_bar.config(text="Genre prediction failed")
            
            # Direct user to train the model if it seems to be the issue
            if "no attribute 'models'" in str(e) or "Ran out of input" in str(e):
                follow_up = messagebox.askyesno("Train Model", 
                                              "It appears the model needs to be trained. Would you like to go to the model training page?")
                if follow_up:
                    self.show_model_page()
    
    def display_prediction_results(self):
        """Display the prediction results."""
        # Clear results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Create a frame for the results table
        table_frame = tk.Frame(self.results_frame, **FRAME_STYLE)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create table headers
        headers = ["Genre", "Probability"]
        for i, header in enumerate(headers):
            header_label = tk.Label(table_frame, text=header, **LABEL_STYLE)
            header_label.grid(row=0, column=i, padx=10, pady=5, sticky=tk.W)
        
        # Add results to table
        for i, (genre, prob) in enumerate(self.prediction_results):
            # Format probability as percentage
            prob_str = f"{prob:.2%}"
            
            # Genre name
            genre_label = tk.Label(table_frame, text=genre, **LABEL_STYLE)
            genre_label.grid(row=i+1, column=0, padx=10, pady=5, sticky=tk.W)
            
            # Probability with progress bar
            prob_frame = tk.Frame(table_frame, **FRAME_STYLE)
            prob_frame.grid(row=i+1, column=1, padx=10, pady=5, sticky=tk.W)
            
            # Create a canvas for the progress bar
            canvas = tk.Canvas(prob_frame, width=200, height=20, bg="white", highlightthickness=0)
            canvas.pack(side=tk.LEFT)
            
            # Draw the progress bar
            bar_width = int(200 * prob)
            canvas.create_rectangle(0, 0, bar_width, 20, fill=PRIMARY_COLOR, outline="")
            
            # Add text label
            prob_label = tk.Label(prob_frame, text=prob_str, **LABEL_STYLE)
            prob_label.pack(side=tk.LEFT, padx=5)
        
        # Add explanation
        explanation_text = """
        The model predicts the probability of each genre based on the movie summary.
        Higher probability indicates stronger confidence in the prediction.
        """
        
        explanation_label = tk.Label(self.results_frame, text=explanation_text, 
                                   justify=tk.LEFT, **LABEL_STYLE)
        explanation_label.pack(pady=10, anchor=tk.W)
    
    def load_prediction_example(self):
        """Load an example summary for prediction."""
        # Sample movie summaries
        examples = [
            "A young boy discovers he is a wizard and attends a magical school where he " 
            "learns spells, makes friends, and confronts a dark wizard who killed his parents.",
            
            "In a post-apocalyptic world, a lone survivor battles against zombies while " 
            "searching for a safe haven and potential cure for the virus that has decimated humanity.",
            
            "Two star-crossed lovers from rival families meet at a party and fall in love, " 
            "leading to a series of tragic events as they try to be together despite their families' feud.",
            
            "A team of skilled thieves plan the perfect heist to rob a heavily guarded bank, " 
            "facing obstacles and betrayals along the way as they attempt to pull off the impossible.",
            
            "On a remote space station, the crew discovers an alien life form that begins " 
            "hunting them one by one, leading to a desperate fight for survival."
        ]
        
        # Choose a random example
        import random
        example = random.choice(examples)
        
        # Set the text
        self.prediction_text.delete("1.0", tk.END)
        self.prediction_text.insert("1.0", example)
    
    def prediction_to_audio(self):
        """Convert the prediction summary to audio."""
        # Check if summary exists
        summary = self.prediction_text.get("1.0", tk.END).strip()
        
        if not summary:
            messagebox.showwarning("Empty Summary", "There is no summary to convert.")
            return
        
        # Show audio page
        self.show_audio_page()
        
        # Set the summary
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", summary)
    
    def save_prediction(self):
        """Save the prediction results."""
        # Check if prediction exists
        if not hasattr(self, 'prediction_results'):
            messagebox.showwarning("No Prediction", "No prediction has been made yet.")
            return
        
        # Get summary
        summary = self.prediction_text.get("1.0", tk.END).strip()
        
        # Get genres
        genres = [genre for genre, _ in self.prediction_results]
        
        # Save the prediction
        try:
            summary_id = save_user_summary(summary, genres)
            
            # Show success message
            messagebox.showinfo("Prediction Saved", 
                               f"Prediction saved successfully with ID: {summary_id}")
        except Exception as e:
            # Handle errors
            error_message = f"Error saving prediction: {str(e)}"
            messagebox.showerror("Save Error", error_message)
