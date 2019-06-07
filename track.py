# track.py
#
# This entry point is useful for testing and debugging the video scanner.
#
# Copyright 2019 Jack Boyce (jboyce@gmail.com)

import os

from hawkeye.tracker import VideoScanner, play_video


if __name__ == '__main__':

    # _filename = 'movies/juggling_test_5.mov'
    _filename = 'movies/TBTB3_9balls.mov'
    # _filename = 'movies/GOPR0622.MP4'
    # _filename = 'movies/GOPR0493.MP4'
    # _scanvideo = 'movies/__Hawkeye__/juggling_test_5_640x480.mp4'
    _scanvideo = 'movies/__Hawkeye__/TBTB3_9balls_640x480.mp4'
    # _scanvideo = 'movies/__Hawkeye__/GOPR0622_640x480.mp4'
    # _scanvideo = 'movies/__Hawkeye__/GOPR0493_640x480.mp4'
    # _scanvideo = None

    watch_video = False

    # print(cv2.getBuildInformation())

    if watch_video:
        notes_step = 4
        start_frame = 730

        if 1 <= notes_step <= 6:
            _filepath = os.path.abspath(_filename)
            _dirname = os.path.dirname(_filepath)
            _hawkeye_dir = os.path.join(_dirname, '__Hawkeye__')
            _basename = os.path.basename(_filepath)
            _basename_noext = os.path.splitext(_basename)[0]

            if notes_step == 6:
                _basename_notes = _basename_noext + '_notes.pkl'
            else:
                _basename_notes = _basename_noext + '_notes{}.pkl'.format(
                                                            notes_step)
            _filepath_notes = os.path.join(_hawkeye_dir, _basename_notes)

            mynotes = VideoScanner.read_notes(_filepath_notes)
        else:
            mynotes = None

        print('Press Q to quit, I to toggle arc labels, '
              'any other key to advance a frame')
        play_video(_filename, notes=mynotes, outfilename=None,
                   startframe=start_frame, keywait=True)
    else:
        startstep = 1
        endstep = 2
        verbosity = 2

        scanner = VideoScanner(_filename, scanvideo=_scanvideo)
        scanner.process(steps=(startstep, endstep), readnotes=True,
                        writenotes=True, notesdir='__Hawkeye__',
                        verbosity=verbosity)
