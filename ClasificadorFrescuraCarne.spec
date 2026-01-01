# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app_multimodal_ux_pro.py'],
    pathex=['D:\\UNA-PUNO-JHON\\tesis3'],
    binaries=[],
    datas=[
        ('runs_multimodal', 'runs_multimodal'),
        ('dataset_full.csv', '.'),
        ('dataset', 'dataset')
    ],
    hiddenimports=[
        'torch',
        'torchvision',
        'cv2',
        'matplotlib',
        'numpy',
        'pandas',
        'PIL',
        'tkinter'
    ],
    excludes=[
        'torch.testing',
        'expecttest',
        'pytest',
        'scipy.tests'
    ],
    noarchive=True,     # 🔥 CLAVE
    optimize=0
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ClasificadorFrescuraCarne',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon='icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='ClasificadorFrescuraCarne'
)
