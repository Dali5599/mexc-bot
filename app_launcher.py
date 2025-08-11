import os, sys, threading, time, webbrowser
from dotenv import load_dotenv

def main():
    # Ensure working directory is the app folder (next to templates/static/etc.)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)

    # Start Uvicorn programmatically
    import uvicorn
    def _run():
        uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="info", reload=False, workers=1)
    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Wait a moment for the server, then open the browser
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000", new=1)

    # Keep the launcher process alive while server thread is running
    try:
        while t.is_alive():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
