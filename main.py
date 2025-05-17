"""
Main module for the Movie Summary Analysis application.
"""

import os
import tkinter as tk
import sys
from gui import MovieSummaryApp
from setup_nltk import setup_nltk

def main():
    """Main entry point for the application."""
    try:
        # Initialize NLTK resources
        print("Setting up NLTK resources...")
        setup_nltk()
        print("NLTK setup complete!")
        
        # Create and configure root window
        root = tk.Tk()
        root.title("Movie Summary Analysis")
        
        # Create application
        app = MovieSummaryApp(root)
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Run the application
        root.mainloop()
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
