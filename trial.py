import os
import sys

# Set environment variables to prevent PyTorch module issues
os.environ["STREAMLIT_WORKAROUNDS_TORCH"] = "1"
os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

# Completely disable module watching for Streamlit
# This is the key setting to prevent the torch._classes/__path__ issue
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

# Additional settings to minimize module scanning issues
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false" 
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Run streamlit with command line arguments
if __name__ == "__main__":
    # Import streamlit after setting environment variables
    import streamlit.web.cli as streamlit_cli
    
    # Set command line arguments without the invalid option
    sys.argv = [
        "streamlit", 
        "run", 
        "streamlit_ui.py", 
        "--server.enableCORS=false", 
        "--server.enableXsrfProtection=false"
    ]
    sys.exit(streamlit_cli.main())
