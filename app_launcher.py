import os, threading, time, webbrowser
from dotenv import load_dotenv

def main():
    # شغّل من مجلد البرنامج نفسه
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)

    import uvicorn
    # 👈 استيراد مباشر يضمن تضمين app.py داخل الـ EXE
    import app as app_module

    def _run():
        # مرّر الكائن مباشرة بدل "app:app"
        uvicorn.run(app_module.app, host="127.0.0.1", port=8000, log_level="info")

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000", new=1)

    try:
        while t.is_alive():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

