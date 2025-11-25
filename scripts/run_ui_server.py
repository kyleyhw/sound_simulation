import uvicorn
import os

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the main application
    # This assumes the script is in 'scripts/' and the app is in 'src/acoustic_system/app'
    app_path = os.path.join(current_dir, '..', 'src')
    
    # Add the src directory to the Python path to allow for correct imports
    import sys
    sys.path.insert(0, app_path)

    # Note: We use a string "acoustic_system.app.main:socket_app" to allow uvicorn's hot-reloading.
    # Uvicorn will look for a variable named 'socket_app' in the 'acoustic_system.app.main' module.
    uvicorn.run("acoustic_system.app.main:socket_app", host="127.0.0.1", port=8000, reload=True)
