# -*- mode: python -*-

block_cipher = None

import os
hawkeye_root = os.path.abspath(os.path.join(SPECPATH, '../..'))

a = Analysis(['../../Hawkeye.py'],
             pathex=[hawkeye_root],
             binaries=[('../../hawkeye/resources/osx/ffmpeg', '.'),
                       ('../../hawkeye/resources/osx/ffprobe', '.'),
                       ('../../hawkeye/resources/osx/libavdevice.58.dylib', '.'),
                       ('../../hawkeye/resources/osx/libavfilter.7.dylib', '.'),
                       ('../../hawkeye/resources/osx/libavformat.57.dylib', '.'),
                       ('../../hawkeye/resources/osx/libavresample.4.dylib', '.'),
                       ('../../hawkeye/resources/osx/libpostproc.55.dylib', '.')
                      ],
             datas=[('../../hawkeye/resources/about_icon.png', './resources'),
                    ('../../hawkeye/resources/preferences_icon.png', './resources'),
                    ('../../hawkeye/resources/busy_icon.gif', './resources'),
                    ('../../hawkeye/resources/yolo-classes.txt', './resources'),
                    ('../../hawkeye/resources/yolov2-tiny.cfg', './resources'),
                    ('../../hawkeye/resources/yolov2-tiny.weights', './resources'),
                    ('../../hawkeye/resources/billiard-balls.wav', './resources')
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
