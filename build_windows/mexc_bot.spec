# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hidden = []
# ensure dynamic imports are included
hidden += collect_submodules('ccxt')
hidden += collect_submodules('fastapi')
hidden += collect_submodules('uvicorn')
hidden += collect_submodules('jinja2')
hidden += collect_submodules('pandas')
hidden += collect_submodules('numpy')
hidden += collect_submodules('dotenv')
hidden += collect_submodules('tenacity')

datas = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('locales', 'locales'),
    ('config.yaml', '.'),
    ('.env.example', '.'),
    ('README_UI.md', '.'),
    ('README_ar.md', '.'),
]

a = Analysis(
    ['app_launcher.py'],
    pathex=[os.getcwd()],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MEXC Bot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,   # keep console for logs
    disable_windowed_traceback=False,
    target_arch=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MEXC Bot'
)
