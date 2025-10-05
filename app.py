"""
Streamlit web application entry point for AI-Powered PDF Organizer.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run the Streamlit app
from app.streamlit_app import main

if __name__ == "__main__":
    main()