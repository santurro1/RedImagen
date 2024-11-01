# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['imageToPattern.py'],
             pathex=['.'],
             binaries=[],
             datas=[('imagen1.jpg', '.'), 
                    ('imagen2.jpg', '.'), 
                    ('imagen3.jpg', '.'), 
                    ('cropped', 'cropped')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
             
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='imageToPattern',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
