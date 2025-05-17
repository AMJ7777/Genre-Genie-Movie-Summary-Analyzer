import nltk
import os
import ssl
import sys
import argparse

def download_nltk_resource(package, nltk_data_dir):
    """
    Download a specific NLTK resource.
    
    Args:
        package (str): The name of the NLTK package to download
        nltk_data_dir (str): The directory to download the package to
    """
    try:
        print(f"\nDownloading {package}...")
        nltk.download(package, download_dir=nltk_data_dir)
        print(f"{package} was downloaded successfully")
    except Exception as e:
        print(f"Error downloading {package}: {str(e)}")
        raise

def setup_nltk():
    """
    Setup NLTK resources needed for the application.
    Downloads resources to the project directory if they don't exist.
    """
    # Fix SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Define the project's NLTK data directory
    project_dir = os.getcwd()
    nltk_data_dir = os.path.join(project_dir, 'nltk_data')
    
    # Create the directory if it doesn't exist
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Add the project's NLTK data directory to the search path (at the beginning)
    nltk.data.path.insert(0, nltk_data_dir)
    
    # List of required NLTK resources
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('omw-1.4', 'corpora/omw-1.4')  # Often needed with wordnet
    ]
    
    # Print the current NLTK data paths for debugging
    print("\nCurrent NLTK data search paths:")
    for path in nltk.data.path:
        print(f"- {path}")
        if os.path.exists(path):
            print(f"  (exists)")
        else:
            print(f"  (does not exist)")
    
    # Download and verify each resource
    for package, path in resources:
        print(f"\nSetting up {package}...")
        try:
            # First try to remove existing package if it's incomplete
            pkg_dir = None
            if 'tokenizers' in path:
                pkg_dir = os.path.join(nltk_data_dir, 'tokenizers', package)
            elif 'corpora' in path:
                pkg_dir = os.path.join(nltk_data_dir, 'corpora', package)
            
            if pkg_dir and os.path.exists(pkg_dir):
                print(f"Removing existing {package} directory...")
                import shutil
                shutil.rmtree(pkg_dir)
            
            # Special handling for wordnet
            if package == 'wordnet':
                print("Special handling for wordnet...")
                # First try downloading wordnet directly
                print("Downloading wordnet...")
                nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
                
                # Verify wordnet installation
                wordnet_dir = os.path.join(nltk_data_dir, 'corpora', 'wordnet')
                if not os.path.exists(wordnet_dir):
                    print("Wordnet directory not found, attempting alternative download...")
                    # Try downloading with force=True
                    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True, force=True)
                
                # Verify lexnames file exists
                lexnames_path = os.path.join(wordnet_dir, 'lexnames')
                if not os.path.exists(lexnames_path):
                    print("Lexnames file missing, attempting to download wordnet again...")
                    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True, force=True)
                
                # Final verification
                if not os.path.exists(lexnames_path):
                    raise LookupError("Failed to download wordnet properly after multiple attempts")
                
                print("Wordnet setup completed successfully")
                continue
            
            # For other packages
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)
            
            # Verify the download was successful
            try:
                nltk.data.find(path)
                print(f"{package} was downloaded and verified successfully")
            except LookupError as e:
                print(f"Failed to verify {package} download: {e}")
                raise
                
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
            raise
    
    print("\nAll NLTK resources are set up correctly!")
    
    # Print the final NLTK data paths for debugging
    print("\nFinal NLTK data search paths:")
    for path in nltk.data.path:
        print(f"- {path}")

def list_available_corpora():
    """List all available NLTK corpora."""
    print("\nAvailable NLTK corpora:")
    for corpus in nltk.corpus.corpus_list():
        print(f"- {corpus}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLTK Setup and Management Tool')
    parser.add_argument('--list', action='store_true', help='List all available NLTK corpora')
    parser.add_argument('--download', type=str, help='Download a specific NLTK package')
    args = parser.parse_args()

    print(f"Python version: {sys.version}")
    print(f"NLTK version: {nltk.__version__}")

    if args.list:
        list_available_corpora()
    elif args.download:
        project_dir = os.getcwd()
        nltk_data_dir = os.path.join(project_dir, 'nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.insert(0, nltk_data_dir)
        download_nltk_resource(args.download, nltk_data_dir)
    else:
        setup_nltk()