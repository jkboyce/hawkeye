# -*- mode: python -*-

block_cipher = None

import os
hawkeye_root = os.path.abspath(os.path.join(SPECPATH, '../..'))

a = Analysis(['../../Hawkeye.py'],
             pathex=[hawkeye_root],
             binaries=[('ffmpeg.exe', '.'),
                       ('ffprobe.exe', '.'),
                       ('avcodec-57.dll', '.'),
                       ('avdevice-57.dll', '.'),
                       ('avfilter-6.dll', '.'),
                       ('avformat-57.dll', '.'),
                       ('avutil-55.dll', '.'),
                       ('postproc-54.dll', '.'),
                       ('swresample-2.dll', '.'),
                       ('swscale-4.dll', '.'),
                       ('opencv_ffmpeg344_64.dll', '.')
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

# Files that Pyinstaller's dependency checker pulls in but aren't needed, so
# remove them from the build. These total 470 MB.

Hawkeye_excludes = [
                'mfc140u.dll',
                'mkl_avx.dll',
                'mkl_avx2.dll',
                'mkl_avx512.dll',
                'mkl_avx512_mic.dll',
                'mkl_blacs_ilp64.dll',
                'mkl_blacs_intelmpi_ilp64.dll',
                'mkl_blacs_intelmpi_lp64.dll',
                'mkl_blacs_lp64.dll',
                'mkl_blacs_mpich2_ilp64.dll',
                'mkl_blacs_mpich2_lp64.dll',
                'mkl_blacs_msmpi_ilp64.dll',
                'mkl_blacs_msmpi_lp64.dll',
                'mkl_cdft_core.dll',
                'mkl_mc.dll',
                'mkl_mc3.dll',
                'mkl_pgi_thread.dll',
                'mkl_scalapack_ilp64.dll',
                'mkl_scalapack_lp64.dll',
                'mkl_sequential.dll',
                'mkl_tbb_thread.dll',
                'mkl_vml_avx.dll',
                'mkl_vml_avx2.dll',
                'mkl_vml_avx512.dll',
                'mkl_vml_avx512_mic.dll',
                'mkl_vml_cmpt.dll',
                'mkl_vml_def.dll',
                'mkl_vml_mc.dll',
                'mkl_vml_mc2.dll',
                'mkl_vml_mc3.dll',
                'MSVCP140.dll',
                'opencv_datasets341.dll',
                'opencv_dnn_objdetect341.dll',
                'opencv_dpm341.dll',
                'opencv_stereo341.dll',
                'opencv_superres341.dll',
                'opencv_videostab341.dll',
                'opencv_xobjdetect341.dll'
              ]
a.binaries = a.binaries - TOC([(x, None, None) for x in Hawkeye_excludes])

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='Hawkeye')

