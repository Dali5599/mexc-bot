import os, sys, threading, time, webbrowser, traceback, importlib.util
from dotenv import load_dotenv

def load_app_module():
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base, "app.py")
    if os.path.exists(app_path):
        spec = importlib.util.spec_from_file_location("app", app_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    import app as mod
    return mod

def start_server():
    import uvicorn
    app_module = load_app_module()
    uvicorn.run(app_module.app, host="127.0.0.1", port=8000, log_level="info")

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        import shutil; shutil.copyfile(".env.example", ".env")
    try:
        t = threading.Thread(target=start_server, daemon=True)
        t.start()
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000", new=1)
        while t.is_alive():
            time.sleep(0.5)
    except Exception:
        with open("run.log", "a", encoding="utf-8") as f:
            f.write("=== CRASH ===\n" + traceback.format_exc())
        print("CRASH — see run.log"); input("Press Enter to exit...")

if __name__ == "__main__":
    main()

