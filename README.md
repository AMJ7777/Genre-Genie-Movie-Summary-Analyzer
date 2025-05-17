# Genre Genie - Movie Summary Analyzer

Genre Genie is an intelligent movie genre classification system that analyzes movie summaries and predicts their genres using natural language processing and machine learning techniques.

## Features

- Movie genre classification based on plot summaries
- Multi-label genre prediction
- User-friendly GUI interface
- Support for multiple languages
- Audio translation capabilities
- Real-time genre analysis

## Prerequisites

- Python 3.8 or higher
- NLTK
- scikit-learn
- PyQt5
- Other dependencies listed in pyproject.toml

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AMJ7777/Genre-Genie-Movie-Summary-Analyzer.git
cd Genre-Genie-Movie-Summary-Analyzer
```

2. Install the required dependencies:
```bash
pip install -e .
```

3. Download required NLTK data:
```bash
python download_nltk.py
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. Enter a movie summary in the text input area
3. Click "Analyze" to get genre predictions
4. View the results in the output panel

## Project Structure

- `main.py` - Entry point of the application
- `gui.py` - GUI implementation using PyQt5
- `models.py` - Machine learning model implementations
- `preprocessing.py` - Text preprocessing utilities
- `utils.py` - Helper functions
- `translation_audio.py` - Translation and audio features
- `setup_nltk.py` - NLTK setup script
- `download_nltk.py` - NLTK data downloader

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK for natural language processing capabilities
- scikit-learn for machine learning algorithms
- PyQt5 for the GUI framework 
