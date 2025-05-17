import nltk
import os

# Create a project-specific directory for NLTK data
project_nltk_dir = os.path.join(os.getcwd(), 'nltk_data')
print(project_nltk_dir)
if not os.path.exists(project_nltk_dir):
    os.makedirs(project_nltk_dir)

# Set the NLTK data path to include our project directory first
nltk.data.path.insert(0, project_nltk_dir)

# Download required resources to our project directory
nltk.download('punkt', download_dir=project_nltk_dir)
nltk.download('stopwords', download_dir=project_nltk_dir)
nltk.download('wordnet', download_dir=project_nltk_dir)
nltk.download('omw-1.4', download_dir=project_nltk_dir)  # This is often needed with wordnet

print(f"NLTK resources have been downloaded to: {project_nltk_dir}")
print("Verifying downloads...")

# Verify the resources were correctly downloaded
try:
    nltk.data.find('tokenizers/punkt')
    print("✓ punkt verified")
except LookupError:
    print("✗ punkt verification failed")

try:
    nltk.data.find('corpora/stopwords')
    print("✓ stopwords verified")
except LookupError:
    print("✗ stopwords verification failed")

try:
    nltk.data.find('corpora/wordnet')
    print("✓ wordnet verified")
except LookupError:
    print("✗ wordnet verification failed")

print("\nNLTK data paths:")
for path in nltk.data.path:
    print(f"- {path}")