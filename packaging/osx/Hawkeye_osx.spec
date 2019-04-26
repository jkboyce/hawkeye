# -*- mode: python -*-

block_cipher = None

import os
hawkeye_root = os.path.abspath(os.path.join(SPECPATH, '../..'))

a = Analysis(['../../Hawkeye.py'],
             pathex=[hawkeye_root],
             binaries=[('ffmpeg', '.'),
                       ('ffprobe', '.'),
                       ('libavdevice.57.dylib', '.'),
                       ('libavfilter.6.dylib', '.'),
                       ('libpostproc.54.dylib', '.'),
                       ('libxvidcore.4.dylib', '.')
                      ],
             datas=[('../../resources/about_icon.png', './resources'),
                    ('../../resources/preferences_icon.png', './resources'),
                    ('../../resources/busy_icon.gif', './resources'),
                    ('../../resources/yolo-classes.txt', './resources'),
                    ('../../resources/yolov2-tiny.cfg', './resources'),
                    ('../../resources/yolov2-tiny.weights', './resources')
             ],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='Hawkeye',
          debug=False,
          strip=False,
          upx=True,
          console=False )

# This is how we manually remove files from the Pyinstaller build, when the
# dependency checker gets things wrong:
Hawkeye_excludes = [
                'libmkl_avx.dylib',
                ]
a.binaries = a.binaries - TOC([(x, None, None) for x in Hawkeye_excludes])

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='Hawkeye')

app = BUNDLE(coll,
             name='Hawkeye.app',
             icon='Hawkeye.icns',
             bundle_identifier=None,
             info_plist={'NSHighResolutionCapable': 'True'})
