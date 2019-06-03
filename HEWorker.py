# HEWorker.py
#
# Hawkeye worker thread for analyzing and transcoding videos.
#
# Copyright 2019 Jack Boyce (jboyce@gmail.com)

import os
import sys
import io
import time
import subprocess
import platform
import json

from PySide2.QtCore import QObject, QThread, Signal, Slot

from HEVideoScanner import HEVideoScanner, HEScanException


class HEWorker(QObject):
    """
    Worker that processes videos. Processing consists of two parts:

    1. Transcoding the video into a format that supports smooth cueing
    2. Scanning the video file to analyze the juggling and produce the
       `notes` dictionary with object detections, arc parameters, etc.

    Video transcoding (step 1) is done first and reported back to the UI
    thread, so that the user can view the video while scanning is still
    underway.

    The worker also handles requests to extract a clip from a source video.
    See on_extract_clip().

    We put this worker in a separate QObject and use signals and slots to
    communicate with it, so that we can do these time-consuming operations on a
    thread separate from the main event loop. The signal/slot mechanism is a
    thread-safe way to communicate in Qt. Look in HEMainWindow.startWorker() to
    see how this thread is initiated and signals and slots connected.
    """

    # signal output from processing
    #    arg1(str) = video file_id
    #    arg2(str) = processing output (should be appended to any prev. output)
    sig_output = Signal(str, str)

    # progress indicator signal for work-in-process
    #    arg1(str) = video file_id
    #    arg2(int) = step #
    #    arg3(int) = max # of steps
    sig_progress = Signal(str, int, int)

    # signal that processing failed with an error
    #    arg1(str) = video file_id
    #    arg2(str) = error message
    sig_error = Signal(str, str)

    # signal video transcoding task is done
    #    arg1(str) = video file_id
    #    arg2(dict) = fileinfo dictionary with file path names
    #    arg3(int) = resolution (vertical pixels) of converted video
    #    arg4(int) = preliminary version of notes dictionary for video
    #    arg5(bool) = successful completion
    sig_video_done = Signal(str, dict, int, dict, bool)

    # signal juggling analysis task is done
    #    arg1(str) = video file_id
    #    arg2(dict) = notes dictionary
    #    arg3(bool) = successful completion
    sig_analyze_done = Signal(str, dict, bool)

    # signal that video clipping task is done
    #    arg1(bool) = successful completion
    sig_clipping_done = Signal(bool)

    def __init__(self, app=None):
        super().__init__()
        self._app = app
        self._resolution = 0
        self._abort = False

    @Slot(dict)
    def on_new_prefs(self, prefs: dict):
        """
        This slot gets signaled when there is a change to the display
        preferences for output videos. The worker uses these preferences to
        create a video with the given vertical pixel dimension.
        """
        if prefs['resolution'] == 'Actual size':
            self._resolution = 0
        else:
            self._resolution = int(prefs['resolution'])

    @Slot(str)
    def on_process_video(self, file_id: str, analyze: bool):
        """
        Signaled when the worker should process a video to make the version
        suitable for single-frame stepping.

        When `analyze` is true, subsequently analyze the video for juggling
        content.
        """
        if self.abort():
            return

        fileinfo = self.make_product_fileinfo(file_id)
        notes = None

        try:
            # check if the file exists and is readable
            file_path = fileinfo['file_path']
            errorstring = ''
            if not os.path.exists(file_path):
                errorstring = f'File {file_path} does not exist'
            elif not os.path.isfile(file_path):
                errorstring = f'File {file_path} is not a file'
            if errorstring != '':
                self.sig_error.emit(file_id, errorstring)
                raise HEProcessingException()

            # check if a valid notes file already exists, and if so load it
            need_notes = True
            notes_path = fileinfo['notes_path']
            if os.path.isfile(notes_path):
                notes = HEVideoScanner.read_notes(notes_path)
                if notes['version'] == HEVideoScanner.CURRENT_NOTES_VERSION:
                    need_notes = False
                else:
                    # old version of notes file -> delete and create a new one
                    self.sig_output.emit(file_id,
                                         'Notes file is old...deleting\n\n')
                    os.remove(notes_path)

            if need_notes:
                scanner = HEVideoScanner(file_path,
                                         scanvideo=fileinfo['scanvid_path'])
                notes = scanner.notes

                if self.get_video_metadata(fileinfo, notes) != 0:
                    raise HEProcessingException()
                """
                OpenCV-based alternative for line above:
                if self.run_scanner(fileinfo, scanner, steps=(1, 1),
                                    writenotes=False) != 0:
                    self.sig_error.emit(file_id,
                                        'Error getting video metadata')
                    raise HEProcessingException()
                """

            if not self.abort():
                # check if the target directory exists, and if not create it
                hawkeye_dir = fileinfo['hawkeye_dir']
                if not os.path.exists(hawkeye_dir):
                    self.sig_output.emit(
                        file_id, f'Creating directory {hawkeye_dir}\n\n')
                    os.makedirs(hawkeye_dir)
                    if not os.path.exists(hawkeye_dir):
                        self.sig_error.emit(
                            file_id, f'Error creating directory {hawkeye_dir}')
                        return

                resolution = self.make_display_video(fileinfo, notes)
                self.sig_video_done.emit(file_id, fileinfo, resolution, notes,
                                         resolution >= 0)
        except HEProcessingException:
            self.sig_video_done.emit(file_id, fileinfo, 0, notes, False)
            if analyze:
                self.sig_analyze_done.emit(file_id, notes, False)
            return

        if analyze and not self.abort():
            self.on_analyze_juggling(file_id, notes)

    @Slot(str, dict)
    def on_analyze_juggling(self, file_id: str, notes: dict):
        """
        Signaled when the user wants to analyze the juggling in a video. This
        fills in most of the fields in the `notes` dictionary.
        """
        if self.abort():
            return

        try:
            need_analysis = (notes['step'] < 6)
            first_step = notes['step'] + 1
        except KeyError:
            need_analysis = True
            first_step = 1

        if need_analysis:
            fileinfo = self.make_product_fileinfo(file_id)

            # the following two checks are already done in on_process_video()
            # but do them again in case the filesystem has changed in the
            # interim.

            # check if the file exists and is readable
            file_path = fileinfo['file_path']
            errorstring = ''
            if not os.path.exists(file_path):
                errorstring = f'File {file_path} does not exist'
            elif not os.path.isfile(file_path):
                errorstring = f'File {file_path} is not a file'
            if errorstring != '':
                self.sig_error.emit(file_id, errorstring)
                self.sig_analyze_done.emit(file_id, notes, False)
                return

            # check if the target directory exists, and if not create it
            hawkeye_dir = fileinfo['hawkeye_dir']
            if not os.path.exists(hawkeye_dir):
                self.sig_output.emit(file_id,
                                     f'Creating directory {hawkeye_dir}\n\n')
                os.makedirs(hawkeye_dir)
                if not os.path.exists(hawkeye_dir):
                    self.sig_output.emit(
                            file_id, f'Error creating directory {hawkeye_dir}')
                    self.sig_error.emit(
                            file_id, f'Error creating directory {hawkeye_dir}')
                    self.sig_analyze_done.emit(file_id, notes, False)
                    return

            if self.make_scan_video(fileinfo) != 0:
                self.sig_analyze_done.emit(file_id, notes, False)
                return

            scanner = HEVideoScanner(file_path,
                                     scanvideo=fileinfo['scanvid_path'],
                                     notes=notes)
            if self.run_scanner(fileinfo, scanner, steps=(first_step, 6),
                                writenotes=True) != 0:
                self.sig_analyze_done.emit(file_id, notes, False)
                return

            try:
                os.remove(fileinfo['scanvid_path'])
            except OSError:
                pass

        self.sig_analyze_done.emit(file_id, notes, True)

    @Slot(str, dict, int)
    def on_extract_clip(self, file_id: str, notes: dict, run_num: int):
        """
        Signaled when the user wants to extract a clip from a video.

        Clip the given run number from the video and save it in the same
        directory as the source video.
        """
        if self.abort():
            return

        run_dict = notes['run'][run_num]
        balls = run_dict['balls']
        throws = run_dict['throws']

        # construct absolute path to the clip we will create
        file_path = os.path.abspath(file_id)
        file_dir = os.path.dirname(file_path)
        file_basename = os.path.basename(file_path)
        file_root, file_ext = os.path.splitext(file_basename)
        clip_basename = (f'{file_root}_run{run_num+1:03}'
                         f'_{balls}b_{throws}t.mp4')
        clip_path = os.path.join(file_dir, clip_basename)

        if os.path.isfile(clip_path):
            self.sig_clipping_done.emit(True)   # clip already exists
            return

        # start clip 3 secs before first throw, end 3 secs after last catch
        fps = notes['fps']
        startframe, endframe = run_dict['frame range']
        starttime = max(0.0, startframe / fps - 3.0)
        endtime = min((notes['frame_count'] - 4) / fps, endframe / fps + 3.0)
        duration = endtime - starttime

        sm, ss = divmod(starttime, 60)
        sh, sm = divmod(sm, 60)
        starttime_str = f'{sh:02.0f}:{sm:02.0f}:{ss:02.3f}'
        dm, ds = divmod(duration, 60)
        dh, dm = divmod(dm, 60)
        duration_str = f'{dh:02.0f}:{dm:02.0f}:{ds:02.3f}'

        # run FFmpeg to make the clip. An alternative approach is to use the
        # "copy" codec to avoid re-encoding the streams, but we would have to
        # take care to start on a keyframe. I haven't found an easy way to do
        # that while keeping the audio in sync.
        args = ['-i', file_path,
                '-ss', starttime_str,
                '-t', duration_str,
                '-c:a', 'aac',              # re-encode audio as AAC
                '-c:v', 'libx264',          # re-encode video as H.264/mp4
                '-preset', 'veryfast',
                '-crf', '20',
                clip_path]
        retcode = self.run_ffmpeg(args, None)

        if retcode != 0 or self.abort():
            try:
                os.remove(clip_path)
            except OSError:
                pass
            if not self.abort():
                self.sig_error.emit('', 'Error saving clip: FFmpeg failed'
                                        f' with return code {retcode}')
        self.sig_clipping_done.emit(retcode == 0)

    def abort(self):
        """
        Return True if the user is trying to quit the app.
        """
        if self._abort:
            return True
        self._abort = QThread.currentThread().isInterruptionRequested()
        if self._abort:
            QThread.currentThread().quit()      # stop thread's event loop
        return self._abort

    def make_product_fileinfo(self, file_id):
        """
        Make a dictionary of file paths for all work products, the exception
        being the path to the display video which is added in
        make_display_video() below.
        """
        file_path = os.path.abspath(file_id)
        file_dir = os.path.dirname(file_path)
        file_basename = os.path.basename(file_path)
        file_basename_noext = os.path.splitext(file_basename)[0]
        hawkeye_dir = os.path.join(file_dir, '__Hawkeye__')
        scanvid_basename = file_basename_noext + '_640x480.mp4'
        scanvid_path = os.path.join(hawkeye_dir, scanvid_basename)
        notes_basename = file_basename_noext + '_notes.pkl'
        notes_path = os.path.join(hawkeye_dir, notes_basename)
        csvfile_path = os.path.join(file_dir, file_basename + '.csv')

        result = dict()
        result['file_id'] = file_id
        result['file_path'] = file_path
        result['file_dir'] = file_dir
        result['file_basename'] = file_basename
        result['file_basename_noext'] = file_basename_noext
        result['hawkeye_dir'] = hawkeye_dir
        result['scanvid_basename'] = scanvid_basename
        result['scanvid_path'] = scanvid_path
        result['notes_basename'] = notes_basename
        result['notes_path'] = notes_path
        result['csvfile_path'] = csvfile_path
        return result

    def get_video_metadata(self, fileinfo, notes):
        """
        Run FFprobe to get some source video metadata. In particular we need:

        1. width in pixels (int)
        2. height in pixels (int)
        3. frames per second (float)
        4. frame count (int)

        This function replaces step 1 in HEVideoScanner, which gets this data
        using OpenCV. We use FFprobe here because it may handle a wider
        variety of input formats, and it reports information not accessible
        through the OpenCV API such as display aspect ratio (DAR).

        Returns 0 on success, nonzero on failure. This should signal sig_error
        on every nonzero return value that isn't an abort.
        """
        file_id = fileinfo['file_id']
        file_path = fileinfo['file_path']

        self.sig_output.emit(file_id, 'Getting metadata for video {}...\n'
                                      .format(notes['source']))

        if getattr(sys, 'frozen', False):
            # running in a bundle
            ffprobe_dir = sys._MEIPASS
        else:
            # running in a normal Python environment
            if platform.system() == 'Windows':
                ffprobe_dir = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        'packaging\\windows')
            elif platform.system() == 'Darwin':
                ffprobe_dir = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        'packaging/osx')
            elif platform.system() == 'Linux':
                ffprobe_dir = '/usr/local/bin'   # fill this in JKB
        ffprobe_executable = os.path.join(ffprobe_dir, 'ffprobe')

        args = ['-v', 'error', '-of', 'json', '-select_streams', 'v:0',
                '-show_streams', '-i', file_path]
        args = [ffprobe_executable] + args

        message = 'Running FFprobe with arguments:\n{}\n'.format(
                ' '.join(args))
        self.sig_output.emit(file_id, message)

        try:
            kwargs = {}
            if platform.system() == 'Windows':
                # Python 3.7:
                # CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
                CREATE_NO_WINDOW = 0x08000000
                kwargs['creationflags'] = CREATE_NO_WINDOW

            p = subprocess.Popen(args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, **kwargs)
            outputhandler = io.TextIOWrapper(p.stdout, encoding='utf-8')
            output = ''

            while p.poll() is None:
                time.sleep(0.1)
                for line in outputhandler:
                    output += line
                    if self.abort():
                        p.terminate()
                        p.wait()
                        return 1

            results = f'FFprobe process ended, return code {p.returncode}\n'
            self.sig_output.emit(file_id, results)

            if p.returncode != 0:
                self.sig_output.emit(
                    file_id, '\n####### Error running FFprobe #######\n')
                self.sig_error.emit(file_id, 'Error getting video metadata')
                return p.returncode

            parsed_output = json.loads(output)
            video_metadata = parsed_output['streams'][0]

            width_orig = int(video_metadata['width'])
            height = int(video_metadata['height'])
            framecount = int(video_metadata['nb_frames'])
            fps_raw = str(video_metadata['avg_frame_rate'])
            dar = video_metadata['display_aspect_ratio']

            # The input video might have non-square pixels, which is a problem
            # for us in the visual analysis. Later we will use FFmpeg's scaler
            # to create scan and display videos with square pixels (i.e.
            # SAR=1:1). Here we use the DAR to calculate what the pixel width
            # will be when we scale to square pixels.
            dar_parts = dar.split(':')
            width = height * int(dar_parts[0]) // int(dar_parts[1])

            # the fps response can either be in numeric form like '30' or
            # '59.94', or in rational form like '60000/1001'. In any case we
            # want fps as a float.
            if '/' in fps_raw:
                fps_parts = fps_raw.split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1])
            else:
                fps = float(fps_raw)

            self.sig_output.emit(file_id, f'height = {height}\n')
            self.sig_output.emit(
                    file_id,
                    f'width = {width_orig}, '
                    f'scaling to width = {width} (DAR = {dar})\n')
            self.sig_output.emit(file_id, f'fps = {fps_raw} = {fps}\n')
            self.sig_output.emit(file_id,
                                 f'estimated frame count = {framecount}\n\n')

            # fill in the same notes fields as HEVideoScanner's step 1
            notes['fps'] = fps
            notes['frame_width'] = width
            notes['frame_height'] = height
            notes['frame_count_estimate'] = framecount
            notes['step'] = 1

            return 0
        except subprocess.SubprocessError as err:
            self.sig_output.emit(
                file_id, '\n####### Error running FFprobe #######\n')
            self.sig_output.emit(
                file_id, f'Error message: {err}\n\n\n')
        except KeyError as err:
            self.sig_output.emit(
                file_id, '\n####### Error running FFprobe #######\n')
            self.sig_output.emit(
                file_id, f'Key error accessing returned data: {err}\n\n\n')

        self.sig_error.emit(file_id, 'Error getting video metadata')
        return 1

    def make_display_video(self, fileinfo, notes):
        """
        The video we display in the UI is not the original video, but a version
        transcoded with FFmpeg. We transcode for four reasons:

        1. The video player can't smoothly step backward a frame at a time
           unless every frame is coded as a keyframe. This is rarely the case
           for source video. We use the x264 encoder's keyint=1 option to
           specify the keyframe interval to be a single frame.
        2. For performance reasons we may want to limit the display resolution
           to a maximum value, in which case we want to rescale.
        3. The video player on Windows gives an error when it loads a video
           with no audio track. Fix this by adding a null audio track.
        4. FFmpeg reads more video formats/codecs than QMediaPlayer, so
           transcoding into standard H.264/mp4 allows us to be compatible with
           a wider range of source video formats.

        Returns the resolution of the transcoded video (vertical scanlines, or
        if 0 then identical to the source video). Return value of -1 indicates
        failure.
        """
        file_id = fileinfo['file_id']
        file_path = fileinfo['file_path']
        hawkeye_dir = fileinfo['hawkeye_dir']
        file_basename_noext = fileinfo['file_basename_noext']

        displayvid_resolution = self._resolution if (
            notes is not None and self._resolution < notes['frame_height']
            ) else 0

        if displayvid_resolution == 0:
            displayvid_basename = file_basename_noext + '_keyint1.mp4'
        else:
            displayvid_basename = (file_basename_noext + '_keyint1_' +
                                   str(displayvid_resolution) + '.mp4')
        displayvid_path = os.path.join(hawkeye_dir, displayvid_basename)
        fileinfo['displayvid_basename'] = displayvid_basename
        fileinfo['displayvid_path'] = displayvid_path

        if os.path.isfile(displayvid_path):
            return displayvid_resolution

        self.sig_output.emit(file_id, 'Video conversion starting...\n')

        if displayvid_resolution == 0:
            # FFmpeg args for native resolution
            args = ['-f', 'lavfi',
                    '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-i', file_path, '-shortest', '-c:v', 'libx264', '-preset',
                    'veryfast', '-tune', 'fastdecode', '-crf', '20', '-vf',
                    'scale=trunc(ih*dar):ih,setsar=1,format=yuv420p',
                    '-x264-params', 'keyint=1', '-c:a', 'aac', '-map', '0:a',
                    '-map', '1:v', displayvid_path]
        else:
            # reduced resolution
            args = ['-f', 'lavfi',
                    '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-i', file_path, '-shortest', '-c:v', 'libx264', '-preset',
                    'veryfast', '-tune', 'fastdecode', '-crf', '20', '-vf',
                    'scale=trunc(ih*dar):ih,scale=-2:'
                    + str(displayvid_resolution) + ',setsar=1,format=yuv420p',
                    '-x264-params', 'keyint=1', '-c:a', 'aac', '-map', '0:a',
                    '-map', '1:v', displayvid_path]

        retcode = self.run_ffmpeg(args, file_id)

        if retcode != 0 or self.abort():
            try:
                os.remove(displayvid_path)
            except OSError:
                # print("got exception: {}".format(err))
                pass
            if not self.abort():
                self.sig_output.emit(
                        file_id, '\nError converting video {}'.format(
                                     fileinfo['file_basename']))
                self.sig_error.emit(
                        file_id, 'Error converting video {}'.format(
                                     fileinfo['file_basename']))
            return -1

        return displayvid_resolution

    def make_scan_video(self, fileinfo):
        """
        Create an H.264/mp4 video at 640x480 resolution to use as input for the
        feature detector. We use a fixed resolution because (a) the feature
        detector performs well on videos of this scale, and (b) we may want to
        experiment with a neural network-based feature detector in the future,
        which would need a fixed input dimension.

        Also OpenCV's support for codecs may be more limited than FFmpeg's, so
        transcoding allows us to process any video format FFmpeg can read.

        Lastly we want to ensure the input to the video scanner is video with
        square pixels. Some source video uses non-square pixels so we use
        FFmpeg's scaler to transform it.

        Returns 0 on success, 1 on failure. This must signal sig_error on every
        nonzero return value that isn't an abort.
        """
        file_id = fileinfo['file_id']
        file_path = fileinfo['file_path']
        scanvid_path = fileinfo['scanvid_path']

        if os.path.isfile(scanvid_path):
            return 0                        # video already exists

        args = ['-hide_banner',
                '-i', file_path,
                '-c:v', 'libx264',          # encode with libx264 (H.264)
                '-crf', '20',               # quality factor (high)
                '-vf',                      # video filters:
                'scale=trunc(ih*dar):ih,'   # scale to square pixels
                'scale=-1:480,'             # scale to height 480
                'crop=min(iw\\,640):480,'   # crop to width 640, if needed
                'pad=640:480:(ow-iw)/2:0,'  # pad to width 640, if needed
                'setsar=1',                 # set SAR=1:1 (square pixels)
                '-an',                      # no audio
                scanvid_path]
        retcode = self.run_ffmpeg(args, file_id)

        if retcode != 0 or self.abort():
            try:
                os.remove(scanvid_path)
            except OSError:
                # print("got exception: {}".format(err))
                pass
            if not self.abort():
                self.sig_output.emit(
                        file_id, '\nError converting video {}'.format(
                                     fileinfo['file_basename']))
                self.sig_error.emit(
                        file_id, 'Error converting video {}'.format(
                                     fileinfo['file_basename']))
            return 1

        return 0

    def run_ffmpeg(self, args, file_id):
        """
        Run FFmpeg as a separate process, optionally sending console output
        back to the UI thread.

        Args:
            args(list):
                Argument list for FFmpeg, minus the executable name
            file_id(str):
                Filename for directing FFmpeg console output back to UI thread
                using sig_output signal. If None then output is ignored.
        Returns:
            retcode(int):
                FFmpeg return code
        """
        if getattr(sys, 'frozen', False):
            # running in a bundle
            ffmpeg_dir = sys._MEIPASS
        else:
            # running in a normal Python environment
            if platform.system() == 'Windows':
                ffmpeg_dir = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        'packaging\\windows')
            elif platform.system() == 'Darwin':
                ffmpeg_dir = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        'packaging/osx')
            elif platform.system() == 'Linux':
                ffmpeg_dir = '/usr/local/bin'   # fill this in JKB
        ffmpeg_executable = os.path.join(ffmpeg_dir, 'ffmpeg')

        args = [ffmpeg_executable] + args

        if file_id is not None:
            message = 'Running FFmpeg with arguments:\n{}\n\n'.format(
                    ' '.join(args))
            self.sig_output.emit(file_id, message)

        try:
            kwargs = {}
            if platform.system() == 'Windows':
                # Python 3.7:
                # CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
                CREATE_NO_WINDOW = 0x08000000
                kwargs['creationflags'] = CREATE_NO_WINDOW

            p = subprocess.Popen(args, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, **kwargs)
            outputhandler = io.TextIOWrapper(p.stdout, encoding='utf-8')

            while p.poll() is None:
                time.sleep(0.1)
                for line in outputhandler:
                    if file_id is not None:
                        self.sig_output.emit(file_id, line)
                    if self.abort():
                        p.terminate()
                        p.wait()
                        return 1

            if file_id is not None:
                self.sig_output.emit(file_id, 'FFmpeg process ended, return '
                                              f'code {p.returncode}\n\n')

            return p.returncode
        except subprocess.SubprocessError as err:
            if file_id is not None:
                self.sig_output.emit(
                    file_id, '\n####### Error running FFmpeg #######\n')
                self.sig_output.emit(
                    file_id, "Error message: {}\n\n\n".format(err))
        return 1

    def run_scanner(self, fileinfo, scanner, steps, writenotes):
        """
        Run the video scanner. Capture stdout and send the output to our UI.

        Returns 0 on success, 1 on failure. This must signal sig_error on every
        nonzero return value that isn't an abort.
        """
        file_id = fileinfo['file_id']
        hawkeye_dir = fileinfo['hawkeye_dir']

        # Install our own output handler at sys.stdout to capture printed text
        # from the scanner and send it to our UI thread.
        def output_callback(s):
            self.sig_output.emit(file_id, s)
        sys.stdout = HEOutputHandler(callback=output_callback)

        # Define a callback function to pass in to HEVideoScanner.process()
        # below. Processing takes a long time (seconds to minutes) and the
        # scanner will call this function at irregular intervals.

        def processing_callback(step=0, maxsteps=0):
            # so UI thread can update progress bar:
            self.sig_progress.emit(file_id, step, maxsteps)

            # Release the GIL periodically so that video drawing operations
            # don't get blocked for too long during processing.
            # HEVideoScanner's step 2 isn't a problem because most of that time
            # is spent in OpenCV so the GIL is mostly free. Steps 3+ though are
            # all Python code (and those also happen to correspond to
            # maxsteps==0), so we periodically sleep during those steps.
            if maxsteps == 0:
                time.sleep(0.001)

            if self.abort():
                # raise exception to bail us out of whereever we are in
                # processing:
                raise HEAbortException()

        try:
            scanner.process(steps=steps,
                            writenotes=writenotes, notesdir=hawkeye_dir,
                            callback=processing_callback, verbosity=2)
        except HEScanException as err:
            self.sig_output.emit(file_id,
                                 '\n####### Error during scanning #######\n')
            self.sig_output.emit(file_id,
                                 "Error message: {}\n\n\n".format(err))
            self.sig_error.emit(file_id, f'Error: {err}')
            return 1
        except HEAbortException:
            # worker thread got an abort signal during processing;
            # the scanner writes the notes file as an atomic operation after
            # processing is complete, so no need to delete a partially-written
            # file here.
            return 1
        finally:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

        self.sig_output.emit(file_id, '\n')
        return 0

# -----------------------------------------------------------------------------


class HEOutputHandler(io.StringIO):
    """
    Simple output handler we install at sys.stdout to capture printed output
    and send it through a callback function instead of printing to the console.
    """
    def __init__(self, callback=None):
        super().__init__()
        self._callback = callback

    def write(self, s: str):
        if self._callback is not None:
            self._callback(s)

# -----------------------------------------------------------------------------


class HEProcessingException(Exception):
    def __init__(self, message=None):
        super().__init__(message)

# -----------------------------------------------------------------------------


class HEAbortException(Exception):
    pass

# -----------------------------------------------------------------------------
