"""
Utility functions for the Movie Summary Analysis application.
"""

import os
import json
import random
import string
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
from PIL import Image, ImageTk
import tkinter as tk
import json

# Directory paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_id():
    """Generate a random ID.
    
    Returns:
        Random ID string.
    """
    # Generate random string of letters and digits
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def display_matplotlib_figure(figure, frame):
    """Display a matplotlib figure in a tkinter frame.
    
    Args:
        figure: Matplotlib figure to display.
        frame: Tkinter frame to display in.
        
    Returns:
        Canvas with the figure.
    """
    # Clear frame
    for widget in frame.winfo_children():
        widget.destroy()
    
    # Create canvas
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    return canvas

def create_svg_image(svg_data, size=(24, 24), fill="#FFFFFF"):
    """Create a Tkinter-compatible image from SVG data.
    
    Args:
        svg_data: SVG data string.
        size: Size of the image (width, height).
        fill: Fill color for the SVG.
        
    Returns:
        Tkinter-compatible PhotoImage.
    """
    # Replace fill color in SVG
    svg_data = svg_data.replace('stroke="currentColor"', f'stroke="{fill}"')
    
    # Create a temporary file
    temp_file = os.path.join(DATA_DIR, "temp.svg")
    with open(temp_file, 'w') as f:
        f.write(svg_data)
    
    # Create PIL image from SVG
    from cairosvg import svg2png
    png_data = svg2png(url=temp_file, output_width=size[0], output_height=size[1])
    
    # Create PIL image from PNG data
    pil_image = Image.open(io.BytesIO(png_data))
    
    # Convert to Tkinter-compatible image
    tk_image = ImageTk.PhotoImage(pil_image)
    
    # Remove temporary file
    os.remove(temp_file)
    
    return tk_image

def save_user_summary(summary, genres=None):
    """Save a user-submitted summary and its predicted genres.
    
    Args:
        summary: The summary text.
        genres: List of predicted genres.
        
    Returns:
        ID of the saved summary.
    """
    # Generate a unique ID
    summary_id = generate_id()
    
    # Create summary data
    summary_data = {
        'id': summary_id,
        'summary': summary,
        'genres': genres or []
    }
    
    # Create user data directory if it doesn't exist
    user_data_dir = os.path.join(DATA_DIR, "user_data")
    os.makedirs(user_data_dir, exist_ok=True)
    
    # Save summary to file
    summary_path = os.path.join(user_data_dir, f"{summary_id}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f)
    
    return summary_id

def load_user_summary(summary_id):
    """Load a user-submitted summary.
    
    Args:
        summary_id: ID of the summary to load.
        
    Returns:
        Summary data dictionary.
    """
    # Get path to summary file
    summary_path = os.path.join(DATA_DIR, "user_data", f"{summary_id}.json")
    
    # Load summary data
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    
    return None

def get_all_user_summaries():
    """Get all user-submitted summaries.
    
    Returns:
        List of summary data dictionaries.
    """
    # Get path to user data directory
    user_data_dir = os.path.join(DATA_DIR, "user_data")
    
    # Create directory if it doesn't exist
    os.makedirs(user_data_dir, exist_ok=True)
    
    # Get all summary files
    summary_files = [f for f in os.listdir(user_data_dir) if f.endswith('.json')]
    
    # Load all summaries
    summaries = []
    for summary_file in summary_files:
        summary_path = os.path.join(user_data_dir, summary_file)
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
            summaries.append(summary_data)
    
    return summaries

def create_label_with_tooltip(parent, text, tooltip_text, **kwargs):
    """Create a label with a tooltip.
    
    Args:
        parent: Parent widget.
        text: Text for the label.
        tooltip_text: Text for the tooltip.
        **kwargs: Additional arguments for the label.
        
    Returns:
        Label widget.
    """
    label = tk.Label(parent, text=text, **kwargs)
    
    # Create tooltip functionality
    def show_tooltip(event):
        x, y, _, _ = label.bbox("insert")
        x += label.winfo_rootx() + 25
        y += label.winfo_rooty() + 25
        
        # Create a toplevel window
        tooltip = tk.Toplevel(label)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x}+{y}")
        
        # Create label with tooltip text
        tooltip_label = tk.Label(tooltip, text=tooltip_text, justify=tk.LEFT,
                               background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                               font=("Helvetica", "10", "normal"))
        tooltip_label.pack(ipadx=3, ipady=3)
        
        # Store tooltip for later destruction
        label.tooltip = tooltip
    
    def hide_tooltip(event):
        if hasattr(label, 'tooltip'):
            label.tooltip.destroy()
    
    # Bind events
    label.bind("<Enter>", show_tooltip)
    label.bind("<Leave>", hide_tooltip)
    
    return label
