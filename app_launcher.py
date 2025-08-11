import os, threading, time, webbrowser, traceback
from dotenv import load_dotenv

def start_server():
    import uvicorn
    import app as app_module
    config = uvicorn.Config(app_module.app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    return server.run()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)

    try:
        # شغّل السيرفر في ثريد منفصل
        t = threading.Thread(target=start_server, daemon=True)
        t.start()
        time.sleep(2.0)
        webbrowser.open("http://127.0.0.1:8000", new=1)

        while t.is_alive():
            time.sleep(0.5)
    except Exception as e:
        with open("run.log", "a", encoding="utf-8") as f:
            f.write("=== CRASH ===\n")
            f.write(traceback.format_exc())
        # إبقاء النافذة مفتوحة لقراءة الرسالة
        print("CRASH, see run.log")
        try:
            input("Press Enter to exit...")
        except Exception:
            pass

if __name__ == "__main__":
    main()
