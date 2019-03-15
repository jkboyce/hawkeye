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
             datas=[('../../resources/haarcascade_upperbody.xml', './resources'),
                    ('../../resources/about_icon.png', './resources'),
                    ('../../resources/preferences_icon.png', './resources'),
                    ('../../resources/busy_icon.gif', './resources')
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

# Files that Pyinstaller's dependency checker pulls in but aren't needed, so
# remove them from the build. These total 367 MB.

Hawkeye_excludes = [
                'libmkl_avx.dylib',
                'libmkl_avx512.dylib',
                'libmkl_blacs_mpich_ilp64.dylib',
                'libmkl_blacs_mpich_lp64.dylib',
                'libmkl_cdft_core.dylib',
                'libmkl_intel_ilp64.dylib',
                'libmkl_mc.dylib',
                'libmkl_mc3.dylib',
                'libmkl_scalapack_ilp64.dylib',
                'libmkl_scalapack_lp64.dylib',
                'libmkl_sequential.dylib',
                'libmkl_tbb_thread.dylib',
                'libmkl_vml_avx.dylib',
                'libmkl_vml_avx2.dylib',
                'libmkl_vml_avx512.dylib',
                'libmkl_vml_mc.dylib',
                'libmkl_vml_mc2.dylib',
                'libmkl_vml_mc3.dylib',
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
