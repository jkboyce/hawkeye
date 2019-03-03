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

from PySide2.QtCore import QObject, QThread, Signal, Slot

from HEVideoScanner import HEVideoScanner, HEScanException


class HEWorker(QObject):
    """
    Worker that wakes up periodically and processes any videos that have not
    been done yet. We do this in a separate thread from the main event loop
    since it's a time-consuming operation.

    Processing consists of two parts:
    1. Scanning the video file to analyze the juggling and produce the
       `notes` dictionary
    2. Transcoding the video into a format that supports smooth cueing

    This derives from QObject in order to emit signals and connect slots to
    other signals. The signal/slot mechanism is a thread-safe way to
    communicate in Qt. Look in HEMainWindow.startWorker() to see how this
    thread is initiated and signals and slots connected.
    """

    # progress indicator signal for work-in-process
    #    arg1(str) = video file_id
    #    arg2(int) = step #
    #    arg3(int) = max # of steps
    sig_progress = Signal(str, int, int)

    # signal output from processing
    #    arg1(str) = video file_id
    #    arg2(str) = processing output (should be appended to any prev. output)
    sig_output = Signal(str, str)

    # signal that processing failed with an error
    #    arg1(str) = video file_id
    #    arg2(str) = error message
    sig_error = Signal(str, str)

    # signal that processing is done for a video
    #    arg1(str) = video file_id
    #    arg2(dict) = notes dictionary for video
    #    arg3(dict) = fileinfo dictionary with file path names
    #    arg4(int) = resolution (vertical pixels) of converted video
    sig_done = Signal(str, dict, dict, int)

    # signal that the app should quit
    sig_quit = Signal()

    def __init__(self, app=None):
        super().__init__()
        self._app = app

    @Slot()
    def work(self):
        """
        Worker thread entry point. Periodically wake up and look for videos
        to process.
        """
        QThread.currentThread().setPriority(QThread.LowPriority)
        self._abort = False
        self._resolution = 0
        self._queue = list()

        while not self._abort:
            if self._app is not None:
                # important so this thread can receive signals:
                self._app.processEvents()

            if len(self._queue) == 0:
                time.sleep(0.1)
                continue

            # we have a video in our queue -> start processing
            file_id = self._queue[0]
            fileinfo = self.make_product_fileinfo(file_id)
            file_path = fileinfo['file_path']

            errorstring = ''
            if not os.path.exists(file_path):
                errorstring = f'File {file_path} does not exist'
            elif not os.path.isfile(file_path):
                errorstring = f'File {file_path} is not a file'
            if errorstring != '':
                self.sig_error.emit(file_id, errorstring)
                del self._queue[0]
                continue

            # check if the target subdirectory exists, and if not create it
            hawkeye_dir = fileinfo['hawkeye_dir']
            if not os.path.exists(hawkeye_dir):
                os.makedirs(hawkeye_dir)
                if not os.path.exists(hawkeye_dir):
                    self.sig_error.emit(file_id,
                                        'Error creating directory {}'.format(
                                            hawkeye_dir))
                    del self._queue[0]
                    continue

            notes = self.make_video_notes(fileinfo)
            if self._abort:
                continue

            resolution = self.make_display_video(fileinfo, notes)
            if self._abort:
                continue

            del self._queue[0]
            self.sig_done.emit(file_id, notes, fileinfo, resolution)

        self.sig_quit.emit()    # signals quit to QApplication

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

    def make_video_notes(self, fileinfo):
        """
        Check if the notes file exists, and if not then create it using
        the video scanner. Capture stdout and send the output to our UI.
        """
        file_id = fileinfo['file_id']
        file_path = fileinfo['file_path']
        hawkeye_dir = fileinfo['hawkeye_dir']
        notes_path = fileinfo['notes_path']
        scanvid_path = fileinfo['scanvid_path']

        if os.path.isfile(notes_path):
            notes = HEVideoScanner.read_notes(notes_path)
            if notes['version'] == HEVideoScanner.CURRENT_NOTES_VERSION:
                return notes
            else:
                # old version of notes file -> delete it and create a new one
                os.remove(notes_path)
                del notes

        """
        Create a video at 640x480 resolution to use as input for the feature
        detector. We use a fixed resolution because (a) the feature detector
        performs well on videos of this scale, and (b) we may want to
        experiment with a neural network-based feature detector in the future,
        which would need a fixed input dimension.
        """
        if not os.path.isfile(scanvid_path):
            args = ['-i', file_path, '-c:v', 'libx264', '-crf', '20', '-vf',
                    'scale=-1:480,crop=min(iw\\,640):480,'
                    'pad=640:480:(ow-iw)/2:0,setsar=1',
                    '-an', scanvid_path]
            retcode = self.run_ffmpeg(args, file_id)

            if retcode != 0 or self._abort:
                try:
                    os.remove(scanvid_path)
                except FileNotFoundError:
                    pass

        # if conversion failed, default to original video for scanning step
        if not os.path.isfile(scanvid_path):
            scanvid_path = None

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
            if self._app is not None:
                # important so we process abort signals to this thread during
                # processing:
                self._app.processEvents()
            if self._abort:
                # raise exception to bail us out of whereever we are in
                # processing:
                raise HEAbortException()

        notes = None
        try:
            scanner = HEVideoScanner(file_path, scanvideo=scanvid_path)
            scanner.process(writenotes=True, notesdir=hawkeye_dir,
                            callback=processing_callback, verbosity=2)
            notes = scanner.notes
        except HEScanException as err:
            self.sig_output.emit(file_id,
                                 '\n####### Error during scanning #######\n')
            self.sig_output.emit(file_id,
                                 "Error message: {}\n\n\n".format(err))
        except HEAbortException:
            # worker thread got an abort signal during processing
            pass
        sys.stdout = sys.__stdout__
        self.sig_output.emit(file_id, '\n')

        return notes

    def make_display_video(self, fileinfo, notes):
        """
        The video we display in the UI is not the original video, but a version
        transcoded with FFmpeg. We transcode for three reasons:

        1. The video player can't smoothly step backward a frame at a time
           unless every frame is coded as a keyframe. This is rarely the case
           for source video. We use the x264 encoder's keyint=1 option to
           specify the keyframe interval to be a single frame.
        2. For performance reasons we may want to limit the display resolution
           to a maximum value, in which case we want to rescale.
        3. The video player on Windows gives an error when it loads a video
           with no audio track. Fix this by adding a null audio track.
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
                    '-i', file_path, '-shortest', '-c:v', 'libx264',
                    '-preset', 'veryfast', '-tune', 'fastdecode', '-crf',
                    '20', '-vf', 'format=yuv420p', '-x264-params',
                    'keyint=1', '-c:a', 'aac', '-map', '0:a', '-map', '1:v',
                    displayvid_path]
        else:
            # reduced resolution
            args = ['-f', 'lavfi',
                    '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-i', file_path, '-shortest', '-c:v', 'libx264',
                    '-preset', 'veryfast', '-tune', 'fastdecode', '-crf',
                    '20', '-vf', 'scale=-2:' + str(displayvid_resolution) +
                    ',format=yuv420p', '-x264-params',
                    'keyint=1', '-c:a', 'aac', '-map', '0:a', '-map', '1:v',
                    displayvid_path]

        retcode = self.run_ffmpeg(args, file_id)

        if retcode != 0 or self._abort:
            try:
                os.remove(displayvid_path)
            except FileNotFoundError:
                pass
            if not self._abort:
                self.sig_output.emit(
                        file_id, '\nError converting video {}'.format(
                                     fileinfo['file_basename']))
                self.sig_error.emit(
                        file_id, 'Error converting video {}'.format(
                                     fileinfo['file_basename']))

        return displayvid_resolution

    def run_ffmpeg(self, args, file_id):
        """
        Run FFmpeg to do video conversion in the background.

        Args:
            args(list):
                Argument list for FFmpeg, minus the executable name
            file_id(str):
                Filename for directing FFmpeg console output back to UI thread
                using sig_output signal
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
        message = 'Running FFMPEG with arguments:\n{}\n\n'.format(
                ' '.join(args))
        self.sig_output.emit(file_id, message)

        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        outputhandler = io.TextIOWrapper(p.stdout, encoding='utf-8')
        while p.poll() is None:
            time.sleep(0.1)
            for line in outputhandler:
                self.sig_output.emit(file_id, line)
            if self._app is not None:
                self._app.processEvents()
            if self._abort:
                p.terminate()
                break

        results = f'Process ended, return code {p.returncode}\n\n'
        self.sig_output.emit(file_id, results)

        return p.returncode

    @Slot(str)
    def on_new_work(self, file_id: str):
        """
        This slot gets signaled when there is a new video to process.
        """
        self._queue.append(file_id)

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

    @Slot()
    def on_app_quit(self):
        """
        This slot gets signaled when the user wants to quit the app.
        """
        self._abort = True

# -----------------------------------------------------------------------------


class HEOutputHandler(io.StringIO):
    """
    Simple output handler class to capture printed output and send the text
    through a callback function instead of printing to the console.
    """
    def __init__(self, callback=None):
        super().__init__()
        self._callback = callback

    def write(self, s: str):
        super().write(s)
        if self._callback is not None:
            self._callback(s)

# -----------------------------------------------------------------------------


class HEAbortException(Exception):
    pass
