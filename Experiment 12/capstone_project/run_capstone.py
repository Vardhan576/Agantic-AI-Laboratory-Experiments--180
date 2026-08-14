import os
import uvicorn

def main():
    print("[Capstone Runner] Preparing environments...")
    # Get directory of current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change current working directory to run main.py contextually
    os.chdir(current_dir)
    
    print("[Capstone Runner] Starting server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
