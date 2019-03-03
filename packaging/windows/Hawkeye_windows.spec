# -*- mode: python -*-

block_cipher = None

import os
hawkeye_root = os.path.abspath(os.path.join(SPECPATH, '../..'))

a = Analysis(['../../Hawkeye.py'],
             pathex=[hawkeye_root],
             binaries=[('ffmpeg.exe', '.')
                      ],
             datas=[('../../Hawkeye/haarcascade_upperbody.xml', '.'),
                    ('../../Hawkeye/about_icon.png', '.'),
                    ('../../Hawkeye/preferences_icon.png', '.')
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
          debug=True,
          strip=False,
          upx=True,
          console=True )

"""
Hawkeye_excludes = [
                'libmkl_avx.dylib',
                'libmkl_avx2.dylib',
                'libmkl_avx512.dylib',
                'libmkl_blacs_mpich_ilp64.dylib',
                'libmkl_blacs_mpich_lp64.dylib',
                'libmkl_cdft_core.dylib',
                'libmkl_core.dylib',
                'libmkl_intel_ilp64.dylib',
                'libmkl_intel_lp64.dylib',
                'libmkl_intel_thread.dylib',
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
                'libopencv_aruco.3.3.dylib',
                'libopencv_bgsegm.3.3.dylib',
                'libopencv_bioinspired.3.3.dylib',
                'libopencv_calib3d.3.3.dylib',
                'libopencv_ccalib.3.3.dylib',
                'libopencv_core.3.3.dylib',
                'libopencv_datasets.3.3.dylib',
                'libopencv_dnn.3.3.dylib',
                'libopencv_dpm.3.3.dylib',
                'libopencv_face.3.3.dylib',
                'libopencv_features2d.3.3.dylib',
                'libopencv_flann.3.3.dylib',
                'libopencv_fuzzy.3.3.dylib',
                'libopencv_highgui.3.3.dylib',
                'libopencv_img_hash.3.3.dylib',
                'libopencv_imgcodecs.3.3.dylib',
                'libopencv_imgproc.3.3.dylib',
                'libopencv_line_descriptor.3.3.dylib',
                'libopencv_ml.3.3.dylib',
                'libopencv_objdetect.3.3.dylib',
                'libopencv_optflow.3.3.dylib',
                'libopencv_phase_unwrapping.3.3.dylib',
                'libopencv_photo.3.3.dylib',
                'libopencv_plot.3.3.dylib',
                'libopencv_reg.3.3.dylib',
                'libopencv_rgbd.3.3.dylib',
                'libopencv_saliency.3.3.dylib',
                'libopencv_shape.3.3.dylib',
                'libopencv_stitching.3.3.dylib',
                'libopencv_structured_light.3.3.dylib',
                'libopencv_superres.3.3.dylib',
                'libopencv_surface_matching.3.3.dylib',
                'libopencv_text.3.3.dylib',
                'libopencv_tracking.3.3.dylib',
                'libopencv_video.3.3.dylib',
                'libopencv_videoio.3.3.dylib',
                'libopencv_videostab.3.3.dylib',
                'libopencv_xfeatures2d.3.3.dylib',
                'libopencv_ximgproc.3.3.dylib',
                'libopencv_xobjdetect.3.3.dylib',
                'libopencv_xphoto.3.3.dylib'
              ]
a.binaries = a.binaries - TOC([(x, None, None) for x in Hawkeye_excludes])
"""

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
