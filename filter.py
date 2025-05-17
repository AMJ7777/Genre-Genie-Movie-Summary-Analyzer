import os
import shutil

# Set up project root and filtered output
project_root = '.'  # Adjust if script isn't run from the root
filtered_output = './filtered_project'
max_file_size_mb = 100

# Folders to exclude
excluded_folders = {'__pycache__', '.venv', 'nltk_data', '.git', '.config', 'matplotlib'}
# File extensions to include
included_extensions = {'.py', '.toml', '.txt', '.tsv', '.svg'}

def should_include(file_path):
    if not os.path.isfile(file_path):
        return False
    if os.path.getsize(file_path) > max_file_size_mb * 1024 * 1024:
        return False
    if not os.path.splitext(file_path)[1] in included_extensions:
        return False
    return True

def copy_filtered_files(src_dir, dest_dir):
    for root, dirs, files in os.walk(src_dir):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_folders and not d.startswith('.')]
        rel_root = os.path.relpath(root, src_dir)
        for file in files:
            src_file = os.path.join(root, file)
            if should_include(src_file):
                dest_folder = os.path.join(dest_dir, rel_root)
                os.makedirs(dest_folder, exist_ok=True)
                shutil.copy2(src_file, os.path.join(dest_folder, file))

# Clean and run
if os.path.exists(filtered_output):
    shutil.rmtree(filtered_output)
os.makedirs(filtered_output)

copy_filtered_files(project_root, filtered_output)

print(f"\n✅ Filtered project copied to: {filtered_output}")
