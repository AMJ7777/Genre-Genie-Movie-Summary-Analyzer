"""
Preprocessing module for the Movie Summary Analysis application.
Handles data cleaning, tokenization, and feature extraction.
"""

import os
import re
import string
import json
import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from setup_nltk import setup_nltk

# Initialize NLTK with required resources
setup_nltk()

# Data paths
DATA_DIR = "data"
PLOT_SUMMARIES_PATH = "attached_assets/plot_summaries.txt"
METADATA_PATH = "attached_assets/movie.metadata.tsv"
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
TFIDF_MODEL_PATH = os.path.join(PREPROCESSED_DIR, "tfidf_vectorizer.pkl")
LEMMATIZER_PATH = os.path.join(PREPROCESSED_DIR, "lemmatizer.pkl")

# Create directories if they don't exist
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

class DataPreprocessor:
    """Class for preprocessing movie summaries and metadata."""
    
    def __init__(self, callback=None):
        """Initialize the DataPreprocessor.
        
        Args:
            callback: A function to call with progress updates (0-100).
        """
        self.callback = callback
        
        # Initialize lemmatizer with error handling
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            print(f"Warning: Failed to initialize WordNetLemmatizer: {e}")
            # Define a simple fallback lemmatizer
            class FallbackLemmatizer:
                def lemmatize(self, word):
                    return word
            self.lemmatizer = FallbackLemmatizer()
        
        # Initialize stop words with error handling
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Failed to initialize stopwords: {e}")
            self.stop_words = set()  # Empty set as fallback
        
        self.genres_map = {}  # Map of genre IDs to genre names
        
    def clean_text(self, text):
        """Clean and preprocess text.
        
        Args:
            text: The text to clean.
            
        Returns:
            Cleaned and preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text.
        
        Args:
            text: The text to process.
            
        Returns:
            A list of lemmatized tokens.
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens 
                      if word not in self.stop_words and len(word) > 2]
        
        return lemmatized
    
    def extract_genres_from_metadata(self, metadata_path):
        """Extract genre information from metadata.
        
        Args:
            metadata_path: Path to the metadata file.
            
        Returns:
            Dictionary mapping movie IDs to lists of genres.
        """
        movie_genres = {}
        
        # Read metadata file
        metadata_df = pd.read_csv(metadata_path, sep='\t', header=None, 
                                 names=['movie_id', 'freebase_id', 'title', 'release_date', 
                                        'revenue', 'runtime', 'languages', 'countries', 'genres'])
        
        # Process each row
        total_rows = len(metadata_df)
        for i, row in enumerate(metadata_df.itertuples()):
            if i % 1000 == 0 and self.callback:
                progress = int(i / total_rows * 50)  # 0-50% for metadata processing
                self.callback(progress)
            
            movie_id = str(row.movie_id)
            genres_str = row.genres if pd.notna(row.genres) else "{}"
            
            # Extract genres
            try:
                # Parse genres string
                genres = []
                if genres_str != "{}":
                    # Remove braces and split by commas
                    genres_items = genres_str.strip('{}').split(', ')
                    for item in genres_items:
                        parts = item.split(': ')
                        if len(parts) == 2:
                            genre_id = parts[0].strip('"')
                            genre_name = parts[1].strip('"')
                            genres.append(genre_name)
                            # Store genre ID to name mapping
                            if genre_id not in self.genres_map:
                                self.genres_map[genre_id] = genre_name
                
                if genres:
                    movie_genres[movie_id] = genres
            except Exception as e:
                print(f"Error processing genres for movie {movie_id}: {e}")
        
        if self.callback:
            self.callback(50)  # 50% complete after metadata processing
            
        return movie_genres
    
    def process_plot_summaries(self, summaries_path, movie_genres):
        """Process plot summaries.
        
        Args:
            summaries_path: Path to the plot summaries file.
            movie_genres: Dictionary mapping movie IDs to lists of genres.
            
        Returns:
            DataFrame with processed summaries and genres.
        """
        summaries_data = []
        
        # Read summaries file
        with open(summaries_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process each summary
        total_lines = len(lines)
        for i, line in enumerate(lines):
            if i % 100 == 0 and self.callback:
                progress = 50 + int(i / total_lines * 50)  # 50-100% for summaries processing
                self.callback(progress)
            
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                movie_id, summary = parts
                
                if movie_id in movie_genres:
                    # Clean the summary
                    cleaned_summary = self.clean_text(summary)
                    
                    # Tokenize and lemmatize
                    tokens = self.tokenize_and_lemmatize(cleaned_summary)
                    processed_summary = ' '.join(tokens)
                    
                    # Add to data
                    summaries_data.append({
                        'movie_id': movie_id,
                        'original_summary': summary,
                        'processed_summary': processed_summary,
                        'genres': movie_genres[movie_id]
                    })
        
        if self.callback:
            self.callback(100)  # 100% complete
            
        # Create DataFrame
        df = pd.DataFrame(summaries_data)
        return df
    
    def save_preprocessed_data(self, df, output_path):
        """Save preprocessed data to CSV.
        
        Args:
            df: DataFrame with preprocessed data.
            output_path: Path to save the data.
        """
        df.to_csv(output_path, index=False)
        
    def create_vectorizer(self, df):
        """Create and fit a TF-IDF vectorizer.
        
        Args:
            df: DataFrame with processed summaries.
            
        Returns:
            Fitted TF-IDF vectorizer.
        """
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        
        # Fit the vectorizer
        tfidf_vectorizer.fit(df['processed_summary'])
        
        return tfidf_vectorizer
    
    def save_vectorizer(self, vectorizer, output_path):
        """Save the vectorizer to a file.
        
        Args:
            vectorizer: The fitted vectorizer.
            output_path: Path to save the vectorizer.
        """
        with open(output_path, 'wb') as f:
            pickle.dump(vectorizer, f)
    
    def save_lemmatizer(self, output_path):
        """Save the lemmatizer to a file.
        
        Args:
            output_path: Path to save the lemmatizer.
        """
        with open(output_path, 'wb') as f:
            pickle.dump(self.lemmatizer, f)
    
    def preprocess_data(self, plot_summaries_path=PLOT_SUMMARIES_PATH, 
                        metadata_path=METADATA_PATH, save=True):
        """Preprocess movie data.
        
        Args:
            plot_summaries_path: Path to the plot summaries file.
            metadata_path: Path to the metadata file.
            save: Whether to save the preprocessed data.
            
        Returns:
            DataFrame with preprocessed data.
        """
        # Extract genres from metadata
        movie_genres = self.extract_genres_from_metadata(metadata_path)
        
        # Process plot summaries
        df = self.process_plot_summaries(plot_summaries_path, movie_genres)
        
        if save:
            # Create directories if they don't exist
            os.makedirs(PREPROCESSED_DIR, exist_ok=True)
            
            # Save preprocessed data
            self.save_preprocessed_data(df, os.path.join(PREPROCESSED_DIR, "preprocessed_data.csv"))
            
            # Create and save vectorizer
            vectorizer = self.create_vectorizer(df)
            self.save_vectorizer(vectorizer, TFIDF_MODEL_PATH)
            
            # Save lemmatizer
            self.save_lemmatizer(LEMMATIZER_PATH)
        
        return df
    
    def is_data_preprocessed(self):
        """Check if data has been preprocessed.
        
        Returns:
            True if preprocessed data exists, False otherwise.
        """
        return (os.path.exists(os.path.join(PREPROCESSED_DIR, "preprocessed_data.csv")) and
                os.path.exists(TFIDF_MODEL_PATH) and
                os.path.exists(LEMMATIZER_PATH))
    
    def load_preprocessed_data(self):
        """Load preprocessed data.
        
        Returns:
            DataFrame with preprocessed data.
        """
        return pd.read_csv(os.path.join(PREPROCESSED_DIR, "preprocessed_data.csv"))
    
    def load_vectorizer(self):
        """Load the TF-IDF vectorizer.
        
        Returns:
            Fitted TF-IDF vectorizer.
        """
        with open(TFIDF_MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    
    def load_lemmatizer(self):
        """Load the lemmatizer.
        
        Returns:
            WordNet lemmatizer.
        """
        with open(LEMMATIZER_PATH, 'rb') as f:
            self.lemmatizer = pickle.load(f)
            return self.lemmatizer
    
    def prepare_training_data(self, test_size=0.2, random_state=42):
        """Prepare data for training.
        
        Args:
            test_size: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            X_train, X_test, y_train, y_test, genres_list, vectorizer
        """
        # Load preprocessed data
        df = self.load_preprocessed_data()
        
        # Limit to a smaller subset for faster debugging if needed
        # Uncomment to use a smaller dataset for testing
        # df = df.sample(min(5000, len(df)), random_state=random_state)
        
        # Load vectorizer
        vectorizer = self.load_vectorizer()
        
        # Get list of all unique genres
        all_genres = set()
        for genres in df['genres'].apply(eval):
            all_genres.update(genres)
        genres_list = sorted(list(all_genres))
        
        print(f"Found {len(genres_list)} unique genres")
        
        # Create multi-label encoding
        y = np.zeros((len(df), len(genres_list)))
        for i, genres in enumerate(df['genres'].apply(eval)):
            for genre in genres:
                if genre in genres_list:
                    y[i, genres_list.index(genre)] = 1
        
        # Analyze class distribution
        genre_counts = np.sum(y, axis=0)
        for i, genre in enumerate(genres_list):
            count = genre_counts[i]
            percent = (count / len(df)) * 100
            print(f"Genre '{genre}': {count} examples ({percent:.2f}%)")
        
        # Make sure each genre has at least one positive example
        valid_genres = np.sum(y, axis=0) > 0
        if not np.all(valid_genres):
            print(f"Removing {np.sum(~valid_genres)} genres with no positive examples")
            y = y[:, valid_genres]
            genres_list = [g for i, g in enumerate(genres_list) if valid_genres[i]]
        
        # Vectorize the summaries
        X = vectorizer.transform(df['processed_summary'])
        
        print(f"Final shape: X={X.shape}, y={y.shape}, genres={len(genres_list)}")
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None)
        
        # Log the class distribution in the training set
        genre_counts_train = np.sum(y_train, axis=0)
        print("\nClass distribution in training set:")
        for i, genre in enumerate(genres_list):
            count = genre_counts_train[i]
            percent = (count / len(y_train)) * 100
            has_both = 1 if (count > 0 and count < len(y_train)) else 0
            print(f"Genre '{genre}': {count}/{len(y_train)} examples ({percent:.2f}%) - Has both classes: {has_both}")
        
        return X_train, X_test, y_train, y_test, genres_list, vectorizer
    
    def preprocess_new_summary(self, summary):
        """Preprocess a new summary.
        
        Args:
            summary: The new summary to preprocess.
            
        Returns:
            Preprocessed summary vector.
        """
        # Load lemmatizer and vectorizer if not already loaded
        if not hasattr(self, 'lemmatizer') or self.lemmatizer is None:
            self.load_lemmatizer()
        
        vectorizer = self.load_vectorizer()
        
        # Clean the summary
        cleaned_summary = self.clean_text(summary)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned_summary)
        processed_summary = ' '.join(tokens)
        
        # Vectorize
        summary_vector = vectorizer.transform([processed_summary])
        
        return summary_vector
