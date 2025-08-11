
# واجهة ويب احترافية بثلاث لغات (AR/EN/FR)

## التشغيل
```bash
pip install -r requirements.txt
cp .env.example .env
# عدّل .env (مفاتيح MEXC + LIVE_TRADING)
uvicorn app:app --reload --port 8000
```
ثم افتح: http://localhost:8000

- مبدئيًا اللغة العربية، ويمكن التبديل من الشريط العلوي (العربية / English / Français).
- الواجهة تدعم RTL تلقائيًا للعربية.
- يمكنك بدء/إيقاف البوت من صفحة اللوحة، وتعديل الإعدادات من صفحة الإعدادات، وتشغيل باك-تست من صفحة Backtest.
- المفاتيح تعدّل من ملف `.env` احترامًا للأمان.
