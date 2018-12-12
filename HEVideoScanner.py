# HEVideoScanner.py
#
# Python class to extract features from a juggling video.
#
# Copyright 2018 Jack Boyce (jboyce@gmail.com)

import sys
import os
import pickle
import copy
from math import sqrt, exp, isnan, atan, degrees, sin, cos, floor
from statistics import median, mean

import cv2

from HETypes import Balltag, Ballarc


class HEVideoScanner:
    """
    Python class that uses OpenCV to process juggling video, determining ball
    movements, juggler positions, and other high-level features. A typical use
    of this would be something like:

        scanner = HEVideoScanner('video.mp4')
        scanner.process()
        notes = scanner.notes
        print('found {} runs in video'.format(notes['runs']))

    Scanning occurs in four distinct steps, and optionally you can specify
    which steps to do (default is all), and whether to write results to
    disk after processing.

    The 'notes' dictionary contains all scan results, and the data is recorded
    as follows:

    notes['version']:
        step number of last processing step completed (int)
    notes['source']:
        full path to source video (str)
    notes['scanvideo']:
        full path to scanned video, or None if identical to source video (str)
    notes['scanner_params']:
        parameters to configure the scanner; see defaultScannerParams() for
        format (dict)
    notes['fps']:
        source video frames per second (float)
    notes['frame_width']:
        source video frame width in pixels (int)
    notes['frame_height']:
        source video frame height in pixels (int)
    notes['camera_tilt']:
        inferred camera tilt angle in source video, in radians (float)
    notes['frame_count']:
        count of frames in source video (int)
    notes['meas'][framenum]:
        list of Balltag objects in frame 'framenum' (list)
    notes['body'][framenum]:
        tuple describing torso bounding box
        (body_x, body_y, body_w, body_h, was_detected)
        observed in frame 'framenum', where all values are in pixel units
        and was_detected is a boolean indicating whether the bounding box
        was a direct detection (True) or inferred (False) (tuple)
    notes['arcs']:
        list of Ballarc objects detected in video (list)
    notes['g_px_per_frame_sq']:
        inferred value of g (gravitational constant) in video, in units of
        pixels/frame^2 (float)
    notes['cm_per_pixel']:
        inferred scale of video in juggling plane, in centimeters per pixel
        (float)
    notes['runs']:
        number of runs detected in video (int)
    notes['run'][run_num]:
        run dictionary describing run number run_num (dict)

    The 'run_dict' dictionary for each run is defined as:

    run_dict['balls']:
        objects detected in run (int)
    run_dict['throws']:
        number of throws in run (int)
    run_dict['throw']:
        list of Ballarc objects for throws in run (list)
    run_dict['throws per sec']:
        inferred throws per second for run, or None (float)
    run_dict['frame range']:
        tuple (frame_start, frame_end) of run's extent in source video (tuple
        of ints)
    run_dict['duration']:
        duration in seconds of run (float)
    """

    def __init__(self, filename, scanvideo=None, params=None):
        """
        Initialize the video scanner. This doesn't do any actual processing;
        see the process() method.

        Args:
            filename(string):
                Filename of video to process. May be absolute path, or path
                relative to the executable.
            scanvideo(string, optional):
                Filename of video to do image detection on. This is assumed to
                be a rescaled version of the video in the 'filename' argument,
                with the same frame rate. If provided, the object detector
                in step 1 will use this version of the video and translate
                coordinates to the original.
            params(dict, optional):
                Parameters to configure the scanner. The function
                HEVideoScanner.defaultScannerParams() returns a dict of the
                expected format.
        Returns:
            None
        """
        self.notes = dict()
        self.notes['source'] = os.path.abspath(filename)
        self.notes['scanvideo'] = (os.path.abspath(scanvideo)
                                   if scanvideo is not None else None)
        self.notes['scanner_params'] = (HEVideoScanner.defaultScannerParams()
                                        if params is None else params)
        self.notes['version'] = 0

    def process(self, steps=(1, 4), readnotes=False, writenotes=False,
                notesdir=None, callback=None, verbosity=0):
        """
        Process the video. Processing occurs in four distinct steps. The
        default is to do all processing steps sequentially, but processing may
        be broken up into multiple calls to this method if desired -- see the
        'steps' argument.

        All output is recorded in the self.notes dictionary. Optionally the
        notes dictionary can be read in from disk prior to processing, and/or
        written to disk after processing.

        Args:
            steps((int, int) tuple, optional):
                Starting and finishing step numbers to execute. Default is
                (1, 4), or all steps.
            readnotes(bool, optional):
                Should the notes dictionary be read from disk prior to
                processing.
            writenotes(bool, optional):
                Should the notes dictionary be written to disk after the
                final step of processing.
            notesdir(string, optional):
                Directory for the optional notes files. Can be an absolute
                path, or a path relative to the video file. Default is the
                same directory as the video file. Note: upon writing, if the
                notes directory doesn't exist then it will be created.
            callback(callable, optional):
                A callable with call signature func([int], [int]) that may
                be provided to update the caller on progress. If the
                optional integer arguments are included, they are the step #
                and estimated total # of steps in processing.
            verbosity(int, optional):
                Verbosity level for printing progress to standard output.
                0 = no output, 1 = key steps, 2 = full output. Default is 0.
        Returns:
            None
        """
        self._callback = callback
        self._verbosity = verbosity
        if self._verbosity >= 1:
            print('Video scanner starting...')

        if readnotes or writenotes:
            dirname = os.path.dirname(self.notes['source'])
            basename = os.path.basename(self.notes['source'])
            basename_noext = os.path.splitext(basename)[0]
            if notesdir is None:
                _notesdir = dirname
            elif os.path.isabs(notesdir):
                _notesdir = notesdir
            else:
                _notesdir = os.path.join(dirname, notesdir)

        step_start, step_end = steps
        if step_start in (2, 3, 4):
            if readnotes:
                _notespath = os.path.join(_notesdir, '{}_notes{}.pkl'.format(
                                          basename_noext, step_start - 1))
                self.notes = HEVideoScanner.read_notes(_notespath)
        else:
            step_start = 1

        for step in range(step_start, step_end + 1):
            if step == 1:
                self.detect_objects(display=False)
            elif step == 2:
                self.build_initial_arcs()
            elif step == 3:
                self.EM_optimize()
            elif step == 4:
                self.analyze_juggling()
            self.notes['version'] = step

        if writenotes:
            if step_end in (1, 2, 3):
                _notespath = os.path.join(_notesdir,
                                          '{}_notes{}.pkl'.format(
                                            basename_noext, step_end))
            else:
                _notespath = os.path.join(_notesdir,
                                          '{}_notes.pkl'.format(
                                            basename_noext))
            HEVideoScanner.write_notes(self.notes, _notespath)
        if self._verbosity >= 1:
            print('Video scanner done')

    # --- Step 1: Extract features from video ---------------------------------

    def detect_objects(self, display=False):
        """
        Find coordinates of thrown objects and the juggler's torso in each
        frame of a video, and store them in the self.notes data structure.

        This function is the only one in the scanner that uses OpenCV.

        Args:
            display(bool, optional):
                if True then show video in a window while processing
        Returns:
            None
        """
        notes = self.notes
        scanvideo = self.notes['scanvideo']

        if self._verbosity >= 1:
            print('Object detection starting on video {}...'.format(
                notes['source']))
        cap = cv2.VideoCapture(notes['source'])
        if not cap.isOpened():
            print('Error opening video file')
            exit()      # do something more graceful here! JKB

        if display:
            cv2.namedWindow('Frame')

        fps = cap.get(cv2.CAP_PROP_FPS)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._verbosity >= 2:
            print('width = {}, height = {}, fps = {}'.format(
                framewidth, frameheight, fps))
            print('estimated frame_count = {}'.format(framecount))

        scan_framewidth, scan_frameheight = framewidth, frameheight

        if scanvideo is not None:
            cap.release()
            cap = cv2.VideoCapture(scanvideo)
            if not cap.isOpened():
                print('Error opening scanner video file')
                exit()      # ...and here JKB
            scan_framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            scan_frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self._verbosity >= 2:
                print(f'Switching to transcoded video {scanvideo}')
                print('width = {}, height = {}'.format(
                    scan_framewidth, scan_frameheight))

        scan_scaledown = frameheight / scan_frameheight

        def scan_to_video_coord(scan_x, scan_y):
            orig_cropwidth = frameheight * (scan_framewidth / scan_frameheight)
            orig_padleft = (framewidth - orig_cropwidth) / 2
            orig_x = orig_padleft + scan_x * scan_scaledown
            orig_y = scan_y * scan_scaledown
            return orig_x, orig_y

        notes['fps'] = fps
        notes['frame_width'] = framewidth
        notes['frame_height'] = frameheight
        notes['camera_tilt'] = 0.0
        notes['scanner_params']['min_tags_per_arc'] = (
                notes['scanner_params']['min_tags_per_arc_high_fps']
                if fps >= 29
                else notes['scanner_params']['min_tags_per_arc_low_fps'])
        notes['scanner_params']['max_distance_pixels'] = (
                notes['scanner_params']['max_distance_pixels_480'] *
                frameheight / 480)
        notes['scanner_params']['radius_window'] = (
                notes['scanner_params']['radius_window_high_res']
                if scan_frameheight >= 480
                else notes['scanner_params']['radius_window_low_res'])
        notes['meas'] = dict()
        notes['body'] = dict()

        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByCircularity = True
        params.minCircularity = 0.3
        params.maxCircularity = 1.1
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByArea = True
        params.minArea = (notes['scanner_params']['min_blob_area_high_res']
                          if scan_frameheight >= 480 else
                          notes['scanner_params']['min_blob_area_low_res'])
        params.maxArea = (notes['scanner_params']['max_blob_area_high_res']
                          if scan_frameheight >= 480 else
                          notes['scanner_params']['max_blob_area_low_res'])
        detector = cv2.SimpleBlobDetector_create(params)

        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            base_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            base_dir = os.path.dirname(os.path.realpath(__file__))
        body_cascade_file = os.path.join(base_dir,
                                         'haarcascade_upperbody.xml')
        body_cascade = cv2.CascadeClassifier(body_cascade_file)

        framenum = framereads = 0
        tag_count = 0
        # timelast = time.perf_counter()

        body_average = None
        body_frames_to_average = int(round(
                fps *
                notes['scanner_params']['body_averaging_time_window_secs']))
        body_frames_averaged = 0
        frames_with_no_body = 0

        def center_distance(a, b):
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            dx = (ax + aw / 2) - (bx + bw / 2)
            dy = (ay + ah / 2) - (by + bh / 2)
            return sqrt(dx * dx + dy * dy)

        while cap.isOpened():
            ret, frame = cap.read()
            framereads += 1

            if not ret:
                if self._verbosity >= 2:
                    print('VideoCapture.read() returned False '
                          'on frame read {}'.format(framereads))
                if framereads > framecount:
                    break
                continue

            notes['meas'][framenum] = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bodies = body_cascade.detectMultiScale(gray, 1.3, 6)

            if (len(bodies) > 1 and
                    body_frames_averaged == body_frames_to_average):
                # find the detection that's closest to the moving average and
                # ignore the rest
                bodies = sorted(
                    bodies, key=lambda i: center_distance(i, body_average))
                bodies = bodies[:1]

            for x, y, w, h in bodies:
                frames_with_no_body = 0
                if body_average is None:
                    body_average = (x, y, w, h)
                    body_frames_averaged = 1
                else:
                    body_frames_averaged = min(body_frames_to_average,
                                               body_frames_averaged + 1)
                    temp1 = (body_frames_averaged - 1) / body_frames_averaged
                    temp2 = 1 / body_frames_averaged
                    body_average = (body_average[0] * temp1 + x * temp2,
                                    body_average[1] * temp1 + y * temp2,
                                    body_average[2] * temp1 + w * temp2,
                                    body_average[3] * temp1 + h * temp2)
                break
            else:
                frames_with_no_body += 1

            if (body_average is not None and
                    frames_with_no_body < body_frames_to_average):
                body_x, body_y = scan_to_video_coord(body_average[0],
                                                     body_average[1])
                body_w = body_average[2] * scan_scaledown
                body_h = body_average[3] * scan_scaledown

                notes['body'][framenum] = (body_x, body_y, body_w, body_h,
                                           True)
                if display:
                    x = int(round(body_average[0]))
                    y = int(round(body_average[1]))
                    w = int(round(body_average[2]))
                    h = int(round(body_average[3]))
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2)

            fgmask = fgbg.apply(frame)
            keypoints = detector.detect(fgmask)
            for kp in keypoints:
                # print('frame {}: ({}, {}) radius={:.1f}'.format(framenum,
                # kp.pt[0], kp.pt[1], kp.size))
                tag_total_weight = 1.0

                if body_average is not None:
                    if kp.pt[1] > body_average[1] + body_average[3]:
                        continue    # skip point entirely
                    if kp.pt[1] > body_average[1]:
                        tag_total_weight = exp(2.0 *
                                               (body_average[1] - kp.pt[1])
                                               / body_average[3])

                tag_x, tag_y = scan_to_video_coord(kp.pt[0], kp.pt[1])
                tag_size = kp.size * scan_scaledown
                notes['meas'][framenum].append(
                    Balltag(framenum, tag_x, tag_y, tag_size,
                            tag_total_weight))
                tag_count += 1
                if display:
                    cv2.circle(frame, (int(round(kp.pt[0])),
                               int(round(kp.pt[1]))),
                               int(kp.size), (0, 0, 255), 1)

            if display:
                cv2.imshow('Frame', frame)
                # cv2.imshow('FG Mask MOG 2', fgmask)
                if cv2.waitKey(10) & 0xFF == ord('q'):  # Q on keyboard exits
                    break

            framenum += 1

            if self._callback is not None:
                self._callback(framenum, framecount)

        notes['frame_count'] = framenum
        if self._verbosity >= 2:
            print('actual frame_count = {}'.format(notes['frame_count']))

        cap.release()
        if display:
            cv2.destroyAllWindows()

        if self._verbosity >= 1:
            print(f'Object detection done: {tag_count} detections')

    # --- Step 2: Build initial set of arcs -----------------------------------

    def build_initial_arcs(self):
        """
        Create an initial set of Ballarcs from the measurements. Do this by
        attempting to chain together neighboring measurements into paths with
        the right parabolic shape.

        Args:
            None
        Returns:
            None
        """
        notes = self.notes

        if self._verbosity >= 1:
            print('Build initial arcs starting...')

        # Scan once to get a small number of arcs, to make a preliminary
        # estimate of gravity
        arcs = self.construct_arcs(maxcount=5)
        self.find_global_params(arcs)

        # Scan again to get all arcs
        arcs = self.construct_arcs()
        self.find_global_params(arcs)

        arcs.sort(key=lambda x: x.f_peak)
        for id_, arc in enumerate(arcs, start=1):
            arc.id_ = id_
        notes['arcs'] = arcs

        if self._verbosity >= 1:
            print('Build initial arcs done: {} arcs created'.format(
                len(arcs)))

    def construct_arcs(self, maxcount=None):
        """
        Piece together neighboring measurements to build parabolic arcs.

        We make two passes through this function, once to find a small number
        (5) of high-confidence arcs that we use to estimate gravity. Then a
        second pass finds all arcs.

        Args:
            maxcount(int):
                maximum number of arcs to build
        Returns:
            list of Ballarcs
        """
        notes = self.notes

        if self._verbosity >= 2:
            print('construct_arcs(): building neighbor lists...')
        self.build_neighbor_lists()
        if self._verbosity >= 2:
            print('done')

        arclist = []
        done_making_arcs = False

        # Build a list of all tags that will be the starting points of new
        # arc. Start from the top of the frame and move down.
        tagqueue = []
        for frame in range(notes['frame_count']):
            tagqueue.extend(notes['meas'][frame])
        tagqueue.sort(key=lambda t: t.y)
        for tag in tagqueue:
            tag.done = False

        for tag1 in tagqueue:
            if tag1.done:
                continue
            made_good_arc = False

            for tag2 in tag1.neighbors:
                if tag2.done:
                    continue
                made_bad_arc = False

                # Try to build an arc through points {tag1, tag2}.
                # Maintain a frontier set of tags reachable in one step.
                try:
                    arc = Ballarc()
                    arc.tags = {tag1, tag2}
                    taken_frames = {tag1.frame, tag2.frame}
                    frontier = set(tag1.neighbors) | set(tag2.neighbors)
                    frontier = {
                        t
                        for t in frontier
                        if t.frame not in taken_frames and t.done is False
                    }
                    tag1.weight = {arc: 1.0}
                    tag2.weight = {arc: 1.0}

                    default_cm_per_pixel = (
                            notes['scanner_params']['default_frame_height_cm']
                            / notes['frame_height'])
                    if 'g_px_per_frame_sq' in notes:
                        arc.e = 0.5 * notes['g_px_per_frame_sq']
                    elif 'fps' in notes:
                        arc.e = 0.5 * 980.7 / (default_cm_per_pixel *
                                               (notes['fps'])**2)
                    else:
                        arc.e = 0.5 * 980.7 / (
                            default_cm_per_pixel *
                            notes['scanner_params']['default_fps']**2)

                    # Initialize arc parameters to fit the first two points
                    arc.f_peak = ((tag1.y - tag2.y) / (2.0 * arc.e * (
                        tag2.frame - tag1.frame)) +
                        0.5 * (tag1.frame + tag2.frame))
                    arc.c = tag1.y - arc.e * (tag1.frame - arc.f_peak)**2
                    arc.b = (tag1.x - tag2.x) / (tag1.frame - tag2.frame)
                    arc.a = tag1.x - arc.b * (tag1.frame - arc.f_peak)

                    while True:
                        if len(frontier) == 0:
                            break
                        # Pick the tag in the frontier closest to the arc
                        temp = [(t, arc.get_distance_from_tag(t, notes))
                                for t in frontier]
                        nexttag, dist = min(temp, key=lambda x: x[1])
                        if (dist > notes['scanner_params']
                                        ['max_distance_pixels']):
                            break

                        # Update the frontier and other data structures, then
                        # optionally re-fit the arc including the new point
                        arc.tags.add(nexttag)
                        taken_frames.add(nexttag.frame)
                        frontier |= set(nexttag.neighbors)
                        frontier = {
                            t
                            for t in frontier
                            if t.frame not in taken_frames and t.done is False
                        }
                        nexttag.weight = {arc: 1.0}
                        if (len(arc.tags) >
                                notes['scanner_params']
                                     ['min_tags_to_curve_fit']):
                            self.fit_arcs([arc])

                        if isnan(arc.e) or (arc.e <= 0) or any(
                                arc.get_distance_from_tag(t, notes) >
                                notes['scanner_params']['max_distance_pixels']
                                for t in arc.tags):
                            made_bad_arc = True
                            break

                    # Arc is finished. Decide whether we want to keep it
                    if not made_bad_arc and self.eval_arc(
                            arc, requirepeak=False) == 0:
                        for t in arc.tags:
                            t.done = True
                        arclist.append(arc)
                        made_good_arc = True

                        if maxcount is not None and len(arclist) >= maxcount:
                            done_making_arcs = True

                        if self._verbosity >= 2:
                            print('made arc number {}, f_peak = {}'
                                  ', e = {}'.format(len(arclist), arc.f_peak,
                                                    arc.e))
                    else:
                        for t in arc.tags:
                            t.weight = None

                except RuntimeWarning:
                    made_good_arc = False
                    continue

                if made_good_arc:
                    break  # break out of tag2 loop

            if done_making_arcs:
                break   # break out of tag1 loop

            if self._callback is not None:
                self._callback()

            tag1.done = True    # so we don't visit again in tag2 loop

        if self._verbosity >= 2:
            # Flag the tags that were assigned to arcs:
            for tag in tagqueue:
                tag.done = False
            for arc in arclist:
                for tag in arc.tags:
                    tag.done = True
            print(
                'construct_arcs() done: {} of {} tags attached '
                'to {} arcs'.format(
                    sum(1 for t in tagqueue if t.done is True), len(tagqueue),
                    len(arclist)))

        # Clean up
        for tag in tagqueue:
            del tag.neighbors
            del tag.done

        return arclist

    def build_neighbor_lists(self):
        """
        For each Balltag, build a list of its neighbors. This mapping is used
        for building arcs efficiently.

        Args:
            None
        Returns:
            None
        """
        notes = self.notes

        frame_count = notes['frame_count']
        v_max = None
        if 'g_px_per_frame_sq' in notes:
            v_max = (sqrt(2 * notes['g_px_per_frame_sq'] *
                     notes['frame_height']))

        for frame in range(frame_count):
            for tag in notes['meas'][frame]:
                tag.neighbors = []

                for frame2 in range(
                        max(0, frame -
                            notes['scanner_params']['max_frame_gap_in_arc']
                            - 1),
                        min(frame_count, frame +
                            notes['scanner_params']['max_frame_gap_in_arc']
                            + 2)):
                    if frame2 == frame:
                        continue
                    tag.neighbors.extend(notes['meas'][frame2])

                # sort by velocity needed to get from A to B, with an optional
                # cap on velocity
                def velocity(t):
                    return (sqrt((t.x - tag.x)**2 + (t.y - tag.y)**2) /
                            abs(t.frame - tag.frame))
                temp = sorted([(t, velocity(t)) for t in tag.neighbors],
                              key=lambda t: t[1])
                if v_max is not None:
                    temp = [(t, v) for t, v in temp if v <= v_max]
                tag.neighbors = [t for t, v in temp]

    def fit_arcs(self, arcs):
        """
        Do a weighted least-squares fit of each arc (assumed to be a parabolic
        trajectory) to the measured points.

        Args:
            arcs(list of Ballarc):
                list of Ballarc objects to fit
        Returns:
            None
        """
        notes = self.notes

        c = cos(notes['camera_tilt'])
        s = sin(notes['camera_tilt'])

        for arc in arcs:
            if len(arc.tags) < 3:
                continue

            T0 = T1 = T2 = T3 = T4 = X1 = T1X1 = Y1 = T1Y1 = T2X1 = T2Y1 = 0
            for tag in arc.tags:
                t = tag.frame - arc.f_peak
                x = tag.x
                y = tag.y
                w = tag.weight[arc]
                T0 += w
                T1 += w * t
                T2 += w * t**2
                T3 += w * t**3
                T4 += w * t**4
                X1 += w * x
                T1X1 += w * t * x
                T2X1 += w * t**2 * x
                Y1 += w * y
                T1Y1 += w * t * y
                T2Y1 += w * t**2 * y

            """
            numpy code for the next section:

            Ax = np.array([[T2, T1], [T1, T0]])
            Bx = np.array([[c * T1X1 - s * T1Y1], [c * X1 - s * Y1]])
            X_new = np.dot(np.linalg.inv(Ax), Bx)
            b_new = X_new[0, 0]
            a_new = X_new[1, 0]

            Ay = np.array([[T4, T3, T2], [T3, T2, T1], [T2, T1, T0]])
            By = np.array([[c * T2Y1 + s * T2X1], [c * T1Y1 + s * T1X1],
                           [c * Y1 + s * X1]])
            Y_new = np.dot(np.linalg.inv(Ay), By)
            e_new = Y_new[0, 0]
            d_new = Y_new[1, 0]
            c_new = Y_new[2, 0]
            """

            Ax_det = T0 * T2 - T1**2
            Ay_det = 2*T1*T2*T3 + T0*T2*T4 - T0*T3**2 - T1**2*T4 - T2**3
            if abs(Ax_det) < 1e-3 or abs(Ay_det) < 1e-3:
                continue

            Ax_inv_11 = T0
            Ax_inv_12 = -T1
            Ax_inv_21 = Ax_inv_12
            Ax_inv_22 = T2
            Bx_1 = c * T1X1 - s * T1Y1
            Bx_2 = c * X1 - s * Y1
            b_new = (Ax_inv_11 * Bx_1 + Ax_inv_12 * Bx_2) / Ax_det
            a_new = (Ax_inv_21 * Bx_1 + Ax_inv_22 * Bx_2) / Ax_det

            Ay_inv_11 = T0 * T2 - T1**2
            Ay_inv_12 = T1 * T2 - T0 * T3
            Ay_inv_13 = T1 * T3 - T2**2
            Ay_inv_21 = Ay_inv_12
            Ay_inv_22 = T0 * T4 - T2**2
            Ay_inv_23 = T2 * T3 - T1 * T4
            Ay_inv_31 = Ay_inv_13
            Ay_inv_32 = Ay_inv_23
            Ay_inv_33 = T2 * T4 - T3**2
            By_1 = c * T2Y1 + s * T2X1
            By_2 = c * T1Y1 + s * T1X1
            By_3 = c * Y1 + s * X1
            e_new = (Ay_inv_11*By_1 + Ay_inv_12*By_2 + Ay_inv_13*By_3) / Ay_det
            d_new = (Ay_inv_21*By_1 + Ay_inv_22*By_2 + Ay_inv_23*By_3) / Ay_det
            c_new = (Ay_inv_31*By_1 + Ay_inv_32*By_2 + Ay_inv_33*By_3) / Ay_det

            if (isnan(a_new) or isnan(b_new) or isnan(c_new) or isnan(d_new)
                    or isnan(e_new) or (e_new == 0)):
                continue

            # Adjust the value of f_peak to make the new d parameter = 0
            f_peak_delta = -d_new / (2.0 * e_new)
            arc.f_peak += f_peak_delta
            a_new += b_new * f_peak_delta
            c_new += d_new * f_peak_delta + e_new * f_peak_delta**2
            arc.a = a_new
            arc.b = b_new
            arc.c = c_new
            arc.e = e_new

    def eval_arc(self, arc, requirepeak=False, checkaccel=True):
        """
        Decide whether an arc meets our quality standards.

        Args:
            arc(Ballarc):
                arc to test
            requirepeak(boolean):
                True requires the arc to include tags on either side of the
                arc's peak
            checkaccel(boolean):
                True requires the arc to have an acceleration that lies within
                allowed bounds
        Returns:
            int:
                Zero if arc is good and should be kept, failure code otherwise
        """
        notes = self.notes

        if (isnan(arc.a) or isnan(arc.b) or isnan(arc.c) or isnan(arc.e)
                or isnan(arc.f_peak)):
            return 1

        if requirepeak:
            f_min, f_max = arc.get_tag_range()
            if not (f_min <= arc.f_peak <= f_max):
                return 3

        if checkaccel and not self.is_acceleration_good(arc.e):
            return 4

        close_tag_count = sum(1 for t in arc.tags if
                              arc.get_distance_from_tag(t, notes) <
                              notes['scanner_params']['max_distance_pixels'])
        if close_tag_count < notes['scanner_params']['min_tags_per_arc']:
            return 2

        return 0

    def is_acceleration_good(self, e):
        """
        Decide whether the quadratic component of the arc's y-motion falls
        within the allowed range.

        Args:
            e(float):
                coefficient of frame**2 in arc's motion
        Returns:
            boolean:
                True if acceleration is allowed, False otherwise
        """
        notes = self.notes

        if e < 0:
            return False

        # Criterion is based on how much we know (or can guess) about gravity
        if 'g_px_per_frame_sq' in notes:
            if (2 * e) < ((1 - notes['scanner_params']['g_window']) *
                          notes['g_px_per_frame_sq']):
                return False
            if (2 * e) > ((1 + notes['scanner_params']['g_window']) *
                          notes['g_px_per_frame_sq']):
                return False
        else:
            max_cm_per_pixel = (notes['scanner_params']['max_frame_height_cm']
                                / notes['frame_height'])
            if 'fps' in notes:
                if (2 * e) < 980.7 / (max_cm_per_pixel * (notes['fps'])**2):
                    return False
            else:
                if (2 * e) < 980.7 / (max_cm_per_pixel *
                                      notes['scanner_params']['max_fps']**2):
                    return False

        return True

    def find_global_params(self, arcs):
        """
        Calculate the acceleration of gravity and the physical scale from a set
        of arc measurements, and add them to notes.

        Args:
            arcs(list of Ballarc)
                arcs fitted to measurements
        Returns:
            None
        """
        notes = self.notes

        if len(arcs) == 0:
            return

        most_tagged_arcs = sorted(arcs, key=lambda a: len(a.tags),
                                  reverse=True)
        g_px_per_frame_sq = 2 * median([a.e for a in most_tagged_arcs[:10]])
        notes['g_px_per_frame_sq'] = g_px_per_frame_sq

        if 'fps' in notes:
            fps = notes['fps']
            notes['cm_per_pixel'] = 980.7 / (g_px_per_frame_sq * fps**2)
            if self._verbosity >= 2:
                print('g_px/f^2 = {}, cm/px = {}'.format(
                    notes['g_px_per_frame_sq'], notes['cm_per_pixel']))

    # --- Step 3: Refine arcs with EM algorithm -------------------------------

    def EM_optimize(self):
        """
        Run the Expectation Maximization (EM) algorithm to optimize the set of
        parabolas. This alternates between calculating weights for each tag's
        affiliation with each arc (E step), and using weighted least-squares
        fitting to refine the parabolas (M step). Try to merge and prune out
        bad arcs as we go.

        References for EM algorithm:
        - Moon, T.K., "The Expectation Maximization Algorithm”, IEEE Signal
          Processing Magazine, vol. 13, no. 6, pp. 47–60, November 1996.
        - Ribnick, E. et al, "Detection of Thrown Objects in Indoor and
          Outdoor Scenes", Proceedings of the 2007 IEEE/RSJ International
          Conference on Intelligent Robots and Systems, IROS 2007.

        Args:
            None
        Returns:
            None
        """
        notes = self.notes
        arcs = notes['arcs']
        keep_iterating = True

        if self._verbosity >= 1:
            print('EM optimization starting...')
            arcs_before = len(arcs)

        """
        It's important to do these steps in a certain order. In particular
        we want to do a merge step before we calculate weights, since the
        latter can attach a lot of spurious detection events to arcs and make
        the "obvious" mergers harder to detect. We also want to always follow
        camera tilt estimation by a least-squares fit, to adapt arc parameters
        to the new tilt angle.

        Under most circumstances the first merge and prune steps will do nearly
        all of the work, and the EM steps will make final tweaks.
        """

        while keep_iterating:
            if self._verbosity >= 2:
                print('estimating camera tilt...')
            self.estimate_camera_tilt(arcs)
            if self._verbosity >= 2:
                print('camera tilt = {} degrees'.format(
                                degrees(notes['camera_tilt'])))

            if self._verbosity >= 2:
                print('fitting arcs...')
            self.fit_arcs(arcs)

            keep_iterating = False

            if self._verbosity >= 2:
                print('merging arcs...')
            for arc in arcs:
                arc.done = False
            while self.merge_arcs(arcs):
                keep_iterating = True

            if self._verbosity >= 2:
                print('pruning arcs...')
            while self.prune_arcs(arcs):
                keep_iterating = True

            if self._verbosity >= 2:
                print('calculating weights...')
            self.calculate_weights(arcs)

            if self._verbosity >= 2:
                print('fitting arcs...')
            self.fit_arcs(arcs)

        self.clean_notes()
        arcs.sort(key=lambda x: x.f_peak)
        if self._verbosity >= 1:
            print('EM done: {} arcs before, {} after'.format(arcs_before,
                                                             len(arcs)))

    def calculate_weights(self, arcs):
        """
        For each measured point, calculate a set of normalized weights for each
        arc. This is used for least-squares fitting in the EM algorithm.

        Args:
            arcs(list of Ballarc):
                list of Ballarc objects
        Returns:
            None
        """
        notes = self.notes

        for frame in notes['meas']:
            for tag in notes['meas'][frame]:
                tag.weight = dict()

        for arc in arcs:
            # Tag must be within a certain size range to attach to an arc
            arc_mradius = arc.get_median_tag_radius()
            r_min, r_max = (arc_mradius * (1 - notes['scanner_params']
                                                    ['radius_window']),
                            arc_mradius * (1 + notes['scanner_params']
                                                    ['radius_window']))

            arc.tags = set()
            f_min, f_max = arc.get_frame_range(notes)

            for frame in range(f_min, f_max):
                x, y = arc.get_position(frame, notes)

                for tag in notes['meas'][frame]:
                    if not (r_min <= tag.radius <= r_max):
                        continue
                    distsq_norm = (((x - tag.x)**2 + (y - tag.y)**2) /
                                   notes['scanner_params']['sigmasq'])
                    if distsq_norm < 5.0:
                        tag.weight[arc] = exp(-distsq_norm)

        for frame in notes['meas']:
            for tag in notes['meas'][frame]:
                weight_sum = sum(tag.weight.values())

                if weight_sum > 1e-5:
                    for arc in tag.weight:
                        tag.weight[arc] = (tag.weight[arc] * tag.total_weight /
                                           weight_sum)
                        arc.tags.add(tag)

    def estimate_camera_tilt(self, arcs):
        """
        Estimate how many degrees the video is rotated, based on estimation of
        x- and y-components of gravity. Estimate acceleration in each direction
        with a least-squares fit.

        Args:
            arcs(list of Ballarc):
                list of Ballarc objects in video
        Returns:
            None
        """
        notes = self.notes

        tilt_sum = 0.0
        tilt_count = 0

        for arc in arcs:
            if len(arc.tags) < 3:
                continue

            T0 = T1 = T2 = T3 = T4 = X1 = T1X1 = T2X1 = Y1 = T1Y1 = T2Y1 = 0
            for tag in arc.tags:
                t = tag.frame - arc.f_peak
                x = tag.x
                y = tag.y
                w = tag.weight[arc]
                T0 += w
                T1 += w * t
                T2 += w * t**2
                T3 += w * t**3
                T4 += w * t**4
                X1 += w * x
                T1X1 += w * t * x
                T2X1 += w * t**2 * x
                Y1 += w * y
                T1Y1 += w * t * y
                T2Y1 += w * t**2 * y

            """
            numpy code for the next section:

            A = np.array([[T4, T3, T2], [T3, T2, T1], [T2, T1, T0]])
            A_inv = np.linalg.inv(A)
            B_y = np.array([[T2Y1], [T1Y1], [Y1]])
            coefs_y = np.dot(A_inv, B_y)
            e_y = coefs_y[0, 0]     # acceleration along y direction
            B_x = np.array([[T2X1], [T1X1], [X1]])
            coefs_x = np.dot(A_inv, B_x)
            e_x = coefs_x[0, 0]     # acceleration along x direction
            """

            A_det = 2*T1*T2*T3 + T0*T2*T4 - T0*T3**2 - T1**2*T4 - T2**3
            if abs(A_det) < 1e-3:
                continue

            A_inv_11 = T0 * T2 - T1**2
            A_inv_12 = T1 * T2 - T0 * T3
            A_inv_13 = T1 * T3 - T2**2
            e_y = (A_inv_11*T2Y1 + A_inv_12*T1Y1 + A_inv_13*Y1) / A_det
            e_x = (A_inv_11*T2X1 + A_inv_12*T1X1 + A_inv_13*X1) / A_det

            if self.is_acceleration_good(e_y):
                tilt = atan(e_x / e_y)
                tilt_sum += tilt
                tilt_count += 1

        notes['camera_tilt'] = ((tilt_sum / tilt_count) if tilt_count > 0
                                else 0.0)

    def merge_arcs(self, arcs):
        """
        Find arcs that are duplicates -- where one arc adequately describes the
        tags assigned to another arc -- and merge them. Return after a single
        merger, i.e., call this repeatedly to merge all arcs.

        Args:
            arcs(list of Ballarc):
                list of Ballarc objects to merge
        Returns:
            boolean:
                True if an arc was eliminated, False otherwise
        """
        notes = self.notes

        for arc1 in arcs:
            if arc1.done:
                continue
            if len(arc1.tags) == 0:
                arc1.done = True
                continue

            f_min, f_max = arc1.get_frame_range(notes)

            arc1_mradius = arc1.get_median_tag_radius()
            r_min1, r_max1 = (arc1_mradius * (1 - notes['scanner_params']
                                                       ['radius_window']),
                              arc1_mradius * (1 + notes['scanner_params']
                                                       ['radius_window']))

            taglist1 = [t for t in arc1.tags
                        if (arc1.get_distance_from_tag(t, notes) <
                            notes['scanner_params']['max_distance_pixels']
                            and t.total_weight > 0.95
                            and (r_min1 <= t.radius <= r_max1))]
            if len(taglist1) < 3:
                arc1.done = True
                continue

            for arc2 in arcs:
                if arc2 is arc1:
                    continue
                if len(arc2.tags) == 0:
                    arc2.done = True
                    continue

                f2_min, f2_max = arc2.get_frame_range(notes)
                if f2_max < f_min or f2_min > f_max:
                    continue

                # debug_focus = (arc1.id_ == 32 and arc2.id_ == 39)
                debug_focus = False
                if debug_focus:
                    print('  trying to merge arc1={} and arc2={}'.format(
                            arc1.id_, arc2.id_))

                """
                Try to build a new arc that merges the tags for each of
                {arc1, arc2}. If the new arc adequately fits the combined
                tags then the arcs can be merged.
                """
                arc2_mradius = arc2.get_median_tag_radius()
                r_min2, r_max2 = (arc2_mradius * (1 - notes['scanner_params']
                                                           ['radius_window']),
                                  arc2_mradius * (1 + notes['scanner_params']
                                                           ['radius_window']))

                taglist2 = [t for t in arc2.tags
                            if (arc2.get_distance_from_tag(t, notes) <
                                notes['scanner_params']['max_distance_pixels']
                                and t.total_weight > 0.95
                                and (r_min2 <= t.radius <= r_max2))]
                if len(taglist2) < 3:
                    continue

                new_arc = copy.copy(arc1)
                new_arc.tags = set(taglist1 + taglist2)
                if (len(new_arc.tags) <
                        notes['scanner_params']['min_tags_per_arc']):
                    continue

                if debug_focus:
                    print('  arc1 tags = {}, arc2 tags = {}, '
                          'combined tags = {}'.format(
                            len(taglist1), len(taglist2), len(new_arc.tags)))

                for tag in new_arc.tags:
                    tag.weight[new_arc] = tag.total_weight
                self.fit_arcs([new_arc])

                tags_poorly_fitted = sum(
                        new_arc.get_distance_from_tag(tag, notes) >
                        notes['scanner_params']['max_distance_pixels']
                        for tag in new_arc.tags)

                if debug_focus:
                    arc1_tags = sorted([(t.frame, round(t.x), round(t.y),
                                       arc1.id_) for t in taglist1],
                                       key=lambda x: x[0])
                    print('  arc1 good tags = {}'.format(arc1_tags))
                    arc2_tags = sorted([(t.frame, round(t.x), round(t.y),
                                       arc2.id_) for t in taglist2],
                                       key=lambda x: x[0])
                    print('  arc2 good tags = {}'.format(arc2_tags))
                    print('  tags poorly fitted = {}'.format(
                            tags_poorly_fitted))

                if tags_poorly_fitted > 2:
                    # merging didn't work
                    for tag in new_arc.tags:
                        del tag.weight[new_arc]
                    if debug_focus:
                        print('  # tags poorly fitted = {}...exiting'.format(
                                tags_poorly_fitted))
                        poor_tags1 = [(t.frame, round(t.x), round(t.y),
                                      round(new_arc.get_distance_from_tag(
                                            t, notes)),
                                      arc1.id_) for t in taglist1 if
                                      new_arc.get_distance_from_tag(t, notes) >
                                      notes['scanner_params']
                                           ['max_distance_pixels']]
                        poor_tags2 = [(t.frame, round(t.x), round(t.y),
                                      round(new_arc.get_distance_from_tag(
                                            t, notes)),
                                      arc2.id_) for t in taglist2 if
                                      new_arc.get_distance_from_tag(t, notes) >
                                      notes['scanner_params']
                                           ['max_distance_pixels']]
                        poor_tags = poor_tags1 + poor_tags2
                        poor_tags.sort(key=lambda x: x[0])
                        print(poor_tags)
                    continue

                # arcs can be merged. Remove the second arc, and retain
                # parameters of the new merged arc.
                arcs.remove(arc2)
                for tag in arc2.tags:
                    try:
                        del tag.weight[arc2]
                    except (AttributeError, TypeError, KeyError):
                        pass
                arc1.f_peak = new_arc.f_peak
                arc1.a = new_arc.a
                arc1.b = new_arc.b
                arc1.c = new_arc.c
                arc1.e = new_arc.e
                arc1.tags = new_arc.tags

                if self._verbosity >= 2:
                    f1_min, _ = arc1.get_tag_range()
                    print("  merged arc {} at frame {} "
                          "into arc {} at frame {}".format(
                            arc2.id_, f2_min, arc1.id_, f1_min))
                return True

            arc1.done = True     # mark so we don't revisit

            if self._callback is not None:
                self._callback()

        return False

    def prune_arcs(self, arcs):
        """
        Eliminate arcs that don't meet the quality standard.

        Args:
            arcs(list of Ballarc):
                list of Ballarc objects to prune
        Returns:
            boolean:
                True if an arc was pruned, False otherwise
        """
        notes = self.notes

        for arc in arcs:
            res = self.eval_arc(arc, requirepeak=False)
            if res > 0:
                arcs.remove(arc)
                for tag in arc.tags:
                    try:
                        del tag.weight[arc]
                    except (AttributeError, TypeError, KeyError):
                        pass
                if self._verbosity >= 2:
                    f_min, f_max = arc.get_frame_range(notes)
                    if res == 1:
                        cause = 'numerics'
                    elif res == 2:
                        cause = 'too few close tags'
                    elif res == 3:
                        cause = 'no peak'
                    elif res == 4:
                        cause = 'accel'
                    else:
                        cause = 'unknown reason'
                    print('  removed arc {} starting at frame {}: {}'.format(
                            arc.id_, f_min, cause))
                return True

            if self._callback is not None:
                self._callback()

        return False

    def clean_notes(self):
        """
        Clean up the notes structure. Toss out tags that don't fit to arcs,
        then delete arcs that don't meet the quality standard. Make a final
        assignment of tags to arcs.

        Args:
            None
        Returns:
            None
        """
        notes = self.notes

        if self._verbosity >= 2:
            print('cleaning notes...')

        for frame in notes['meas']:
            for tag in notes['meas'][frame]:
                tag.done = False
        for arc in notes['arcs']:
            arc.done = False

        tags_removed = tags_remaining = arcs_removed = 0
        keep_cleaning = True

        while keep_cleaning:
            for frame in notes['meas']:
                tags_to_kill = []

                for tag in notes['meas'][frame]:
                    if tag.done:
                        continue
                    if not self.is_tag_good(tag):
                        tags_to_kill.append(tag)
                        continue
                    tag.done = True

                for tag in tags_to_kill:
                    notes['meas'][frame].remove(tag)
                    for arc in tag.weight:
                        arc.tags.remove(tag)
                        arc.done = False
                    tags_removed += 1

                if self._callback is not None:
                    self._callback()

            arcs_to_kill = []
            keep_cleaning = False

            for arc in notes['arcs']:
                if arc.done:
                    continue
                if self.eval_arc(arc, requirepeak=True) > 0:
                    arcs_to_kill.append(arc)
                    continue
                arc.done = True

            for arc in arcs_to_kill:
                notes['arcs'].remove(arc)
                for tag in arc.tags:
                    try:
                        del tag.weight[arc]
                    except (AttributeError, TypeError, KeyError):
                        pass
                    tag.done = False
                keep_cleaning = True
                arcs_removed += 1
                if self._verbosity >= 2:
                    f_min, _ = arc.get_frame_range(notes)
                    print('  removed arc {} starting at frame {}'.format(
                            arc.id_, f_min))

        # Final cleanup: Delete unneeded data and make final assignments of
        # tags to arcs.
        for arc in notes['arcs']:
            del arc.done
            arc.tags = set()
        for frame in notes['meas']:
            for tag in notes['meas'][frame]:
                temp = [(arc, arc.get_distance_from_tag(tag, notes))
                        for arc in tag.weight]
                final_arc, _ = min(temp, key=lambda x: x[1])
                tag.arc = final_arc
                final_arc.tags.add(tag)
                del tag.weight
                del tag.done
                tags_remaining += 1

        if self._verbosity >= 2:
            print("cleaning done: {} tags removed, {} tags remaining, "
                  "{} arcs removed".format(tags_removed, tags_remaining,
                                           arcs_removed))

    def is_tag_good(self, tag):
        if tag.arc is not None:
            return True

        if tag.weight is None:
            return False

        notes = self.notes

        return any(arc.get_distance_from_tag(tag, notes) <
                   notes['scanner_params']['max_distance_pixels']
                   for arc in tag.weight)

    # --- Step 4: Analyze juggling patterns -----------------------------------

    def analyze_juggling(self):
        """
        Build out a higher-level description of the juggling using the
        individual throw arcs we found in steps 1-3.

        Args:
            None
        Returns:
            None
        """
        notes = self.notes
        if self._verbosity >= 1:
            print('Juggling analyzer starting...')

        self.add_missing_torso_measurements()
        self.compile_arc_data()

        runs = self.find_runs()
        if self._verbosity >= 2:
            print(f'Number of runs detected = {notes["runs"]}')

        """
        Analyze each run in turn. All run-related information is stored in
        a dictionary called run_dict.
        """
        notes['run'] = list()
        for run_id, run in enumerate(runs, start=1):
            # assign sequence numbers
            for throw_id, arc in enumerate(sorted(
                            run, key=lambda x: x.f_throw), start=1):
                arc.run_id = run_id
                arc.throw_id = throw_id

            run_dict = dict()
            run_dict['throws'] = len(run)
            run_dict['throw'] = run

            if self._verbosity >= 2:
                print(f'--- Analyzing run {run_id} ------------------------')
                print(f'Number of arcs detected = {run_dict["throws"]}')

            self.connect_arcs(run_dict)
            self.assign_hands(run_dict)
            self.estimate_ball_count(run_dict)
            self.analyze_throw_errors(run_dict)

            notes['run'].append(run_dict)

        if self._verbosity >= 2:
            print(f'--------------------------------------------')
        if self._verbosity >= 1:
            print('Juggling analyzer done')

    def add_missing_torso_measurements(self):
        """
        Because the Haar cascade detector is not perfect, the juggler torso
        is often not detected on every frame. Fill in missing measurements
        with an estimate.
        """
        notes = self.notes
        last_torso = None

        for framenum in range(0, notes['frame_count']):
            if framenum in notes['body']:
                last_torso = notes['body'][framenum]
            else:
                """
                Torso was not detected for this frame. Estimate the torso box
                based on tagged ball positions. Find the most extremal tags
                nearby in time.
                """
                f_min = max(framenum - 120, 0)
                f_max = min(framenum + 120, notes['frame_count'])

                nearby_tags = []
                for frame in range(f_min, f_max):
                    nearby_tags.extend(notes['meas'][frame])

                if len(nearby_tags) > 0:
                    """
                    y_sorted_tags = sorted(nearby_tags, key=lambda t: t.y)
                    y_max = median([t.y for t in y_sorted_tags[-5:]])
                    x_sorted_tags = sorted(nearby_tags, key=lambda t: t.x)
                    x_min = median([t.x for t in x_sorted_tags[:5]])
                    x_max = median([t.x for t in x_sorted_tags[-5:]])

                    w = 0.7 * (x_max - x_min)
                    h = 0.8 * w
                    x, y = 0.5 * (x_min + x_max - w), y_max - h
                    """
                    x, y, w, h, _ = last_torso
                    x_sorted_tags = sorted(nearby_tags, key=lambda t: t.x)
                    x_min = median([t.x for t in x_sorted_tags[:5]])
                    x_max = median([t.x for t in x_sorted_tags[-5:]])
                    x = 0.5 * (x_min + x_max - w)

                    notes['body'][framenum] = (x, y, w, h, False)
                    # print(f'added torso to frame {framenum}')

    def compile_arc_data(self):
        """
        Work out some basic information about each arc in the video: Throw
        height, throw position, etc.
        """
        notes = self.notes
        arcs = notes['arcs']

        c = cos(notes['camera_tilt'])
        s = sin(notes['camera_tilt'])

        for arc in arcs:
            # Calculate when each arc was thrown and caught. Assume that the
            # bottom of the torso measurement box represents the throw and
            # catch elevation.
            x, y, w, h, _ = notes['body'][round(arc.f_peak)]

            # (xs_b, ys_b) are the screen coordinates of the bottom center
            # of the torso box
            xs_b = x + 0.5 * w
            ys_b = y + h
            # in rotated (juggler) coordinates:
            x_b = xs_b * c - ys_b * s
            y_b = xs_b * s + ys_b * c

            df2 = (y_b - arc.c) / arc.e
            if df2 > 0:
                df = sqrt(df2)
            else:
                # throw peak is below the bottom of the torso box (rare)
                arc_fmin, arc_fmax = arc.get_tag_range()
                df = max(abs(arc.f_peak - arc_fmin),
                         abs(arc.f_peak - arc_fmax))

            arc.f_throw = arc.f_peak - df
            arc.f_catch = arc.f_peak + df
            arc.x_throw = (arc.a - arc.b * df) - x_b
            arc.x_catch = (arc.a + arc.b * df) - x_b
            arc.height = y_b - arc.c
            arc.x_origin = x_b
            arc.y_origin = y_b

    def find_runs(self):
        """
        Separate arcs into a set of runs, by assuming that two arcs that
        overlap in time are part of the same run.

        Args:
            None
        Returns:
            runs(list):
                List of runs, each of which is a list of Ballarc objects
        """
        notes = self.notes
        arcs = notes['arcs']

        if len(arcs) == 0:
            notes['runs'] = 0
            return []

        runs = list()
        sorted_arcs = sorted(arcs, key=lambda a: a.f_throw)
        first_arc = sorted_arcs[0]
        current_run = [first_arc]
        current_max_frame = first_arc.f_catch

        for arc in sorted_arcs[1:]:
            if arc.f_throw < current_max_frame:
                current_run.append(arc)
                current_max_frame = max(current_max_frame, arc.f_catch)
            else:
                # got a gap in time -> start a new run
                runs.append(current_run)
                current_run = [arc]
                current_max_frame = arc.f_catch
        runs.append(current_run)

        # filter out any runs that are too short
        good_runs, bad_arcs = [], []
        for run in runs:
            good_runs.append(run) if len(run) >= 2 else bad_arcs.extend(run)
        notes['arcs'] = [a for a in notes['arcs'] if a not in bad_arcs]
        for arc in bad_arcs:
            for tag in arc.tags:
                notes['meas'][tag.frame].remove(tag)
        runs = good_runs

        notes['runs'] = len(runs)
        return runs

    def connect_arcs(self, run_dict):
        """
        Try to connect arcs together that represent subsequent throws for
        a given ball. Do this by filling in arc.prev and arc.next for each
        arc in a given run, forming a linked list for each ball in the pattern.

        Since some arcs are not detected (e.g. very low throws), this process
        can often make mistakes.

        Args:
            run_dict(dict):
                dictionary of information for a given run
        Returns:
            None
        """
        notes = self.notes
        run = run_dict['throw']
        f_firstthrow = min(arc.f_throw for arc in run)
        f_lastthrow = max(arc.f_throw for arc in run)
        f_lastcatch = max(arc.f_catch for arc in run)
        run_dict['frame range'] = (f_firstthrow, f_lastcatch)
        run_dict['duration'] = (f_lastcatch - f_firstthrow) / notes['fps']

        if f_lastthrow != f_firstthrow:
            run_dict['throws per sec'] = (
                    (run_dict['throws'] - 1) /
                    ((f_lastthrow - f_firstthrow) / notes['fps']))

            # link together arcs over time that correspond to throws for
            # the same ball. This will allow us to determine the pattern
            # for this run.
            min_dwell_frames = 0.1 * notes['fps']
            likely_dwell_frames = notes['fps'] / run_dict['throws per sec']
            for arc in run:
                possible_nexts = [arc2 for arc2 in run
                                  if (arc2.f_throw >
                                      (arc.f_catch + min_dwell_frames))
                                  and arc2.prev is None]
                if len(possible_nexts) == 0:
                    arc.next = None
                elif len(possible_nexts) == 1:
                    arc.next = possible_nexts[0]
                    possible_nexts[0].prev = arc
                else:
                    likely_nexts = [(arc2, abs(arc2.f_throw - arc.f_catch -
                                               likely_dwell_frames))
                                    for arc2 in possible_nexts]
                    likely_nexts = sorted(likely_nexts, key=lambda i: i[1])
                    """
                    Assume that the next arc is either the first or second
                    element of likely_nexts. If the first is much better in
                    terms of timing then choose it. Otherwise compare
                    throwing and catching positions to break the tie.
                    """
                    if likely_nexts[1][1] > 0.5 * likely_dwell_frames:
                        arc.next = likely_nexts[0][0]
                        likely_nexts[0][0].prev = arc
                    elif (abs(likely_nexts[0][0].x_throw - arc.x_catch) <
                            abs(likely_nexts[1][0].x_throw - arc.x_catch)):
                        arc.next = likely_nexts[0][0]
                        likely_nexts[0][0].prev = arc
                    else:
                        arc.next = likely_nexts[1][0]
                        likely_nexts[1][0].prev = arc
        else:
            # likely just a single throw in the run
            run_dict['throws per sec'] = None
            for arc in run:
                arc.next = arc.prev = None

    def assign_hands(self, run_dict):
        """
        Assign hands to throws for a given run. Do this by filling in
        arc.hand_throw (guaranteed) and arc.hand_catch (where possible) for
        each arc in the run. Assigns None to hand_catch where it is unknown.

        Args:
            run_dict(dict):
                dictionary of information for a given run
        Returns:
            None
        """
        notes = self.notes
        run = run_dict['throw']
        vocal = (self._verbosity >= 2)

        if vocal:
            print('Assigning hands to arcs...')

        # Start by making high-probability assignments of catching hands.
        # We start with catches because they are typically farther from the
        # centerline of the body and therefore more definitively on one side
        # or the other.
        for arc in run:
            # _, _, w, _, _ = notes['body'][round(arc.f_peak)]
            w = 70.0 / notes['cm_per_pixel']

            if arc.x_catch > 0.5 * w:
                arc.hand_catch = 'left'
            elif arc.x_catch < -0.5 * w:
                arc.hand_catch = 'right'
            else:
                arc.hand_catch = None
            if vocal and arc.hand_catch is not None:
                print(f'wide catch; set arc {arc.throw_id} '
                      f'to catch in {arc.hand_catch}')

        # Propagate those assignments to any subsequent throws of same balls:
        for arc in run:
            arc.hand_throw = None if arc.prev is None else arc.prev.hand_catch
            if vocal and arc.hand_throw is not None:
                print(f'propagating; set arc {arc.throw_id} '
                      f'to throw from {arc.hand_throw}')

        """
        Now the main algorithm. Our strategy is to maintain a queue of arcs
        that have had hand_throw assigned. We will try to use these arcs to
        assign hand_throw for nearby arcs that are unassigned, at which point
        they are added to the queue. Continue this process recursively for as
        long as we can.
        """
        arc_queue = [arc for arc in run if arc.hand_throw is not None]

        if len(arc_queue) == 0:
            # Nothing assigned yet; assign something to get started
            arc = max(run, key=lambda a: abs(a.x_throw))
            arc.hand_throw = 'right' if arc.x_throw < 0 else 'left'
            arc_queue = [arc]
            if vocal:
                print(f'no throws assigned; set arc {arc.throw_id} '
                      f'to throw from {arc.hand_throw}')

        # arcs that originate within 0.05s and 10cm of one another are
        # assumed to be a multiplex throw from the same hand:
        mp_window_frames = 0.05 * notes['fps']
        mp_window_pixels = 10.0 / notes['cm_per_pixel']

        # assume that a hand can't make two distinct throws within 0.23s
        # (or less) of each other:
        min_cycle_frames = 0.23 * notes['fps']

        while True:
            while len(arc_queue) > 0:
                assigned_arc = arc_queue.pop()

                """
                Two cases for other arcs that can have throw hand assigned
                based on assigned_arc:
                1. arcs that are very close in time and space, which must
                   be from the same hand (a multiplex throw)
                2. arcs thrown within min_cycle_frames of its throw time,
                   which should be from the opposite hand
                """
                mp_arcs = [arc for arc in run if arc.hand_throw is None
                           and (assigned_arc.f_throw - mp_window_frames) <
                           arc.f_throw <
                           (assigned_arc.f_throw + mp_window_frames)
                           and (assigned_arc.x_throw - mp_window_pixels) <
                           arc.x_throw <
                           (assigned_arc.x_throw + mp_window_pixels)]
                for arc in mp_arcs:
                    arc.hand_throw = assigned_arc.hand_throw
                    arc_queue.append(arc)
                    if vocal:
                        print(f'multiplex throw; set arc {arc.throw_id} '
                              f'to throw from {arc.hand_throw}')

                close_arcs = [arc for arc in run if arc.hand_throw is None
                              and (assigned_arc.f_throw - min_cycle_frames)
                              < arc.f_throw <
                              (assigned_arc.f_throw + min_cycle_frames)]
                for arc in close_arcs:
                    arc.hand_throw = 'right' if (assigned_arc.hand_throw
                                                 == 'left') else 'left'
                    arc_queue.append(arc)
                    if vocal:
                        print(f'close timing; set arc {arc.throw_id} '
                              f'to throw from {arc.hand_throw}')

            # If there are still unassigned throws, find the one that is
            # closest in time to one that is already assigned.
            unassigned_arcs = [arc for arc in run if arc.hand_throw is None]
            if len(unassigned_arcs) == 0:
                break

            assigned_arcs = [arc for arc in run if arc.hand_throw is not None]
            closest_assigned = [(arc, min(assigned_arcs, key=lambda a:
                                          abs(arc.f_throw - a.f_throw)))
                                for arc in unassigned_arcs]
            arc_toassign, arc_assigned = min(closest_assigned,
                                             key=lambda p: abs(p[0].f_throw -
                                                               p[1].f_throw))

            # We want to assign a throw hand to arc_toassign. First
            # check if it's part of a synchronous throw pair, in which
            # case we'll assign hands based on locations.
            sync_arcs = [arc for arc in run if
                         abs(arc.f_throw - arc_toassign.f_throw) <
                         mp_window_frames and arc is not arc_toassign]
            if len(sync_arcs) > 0:
                arc_toassign2 = sync_arcs[0]
                if arc_toassign.x_throw > arc_toassign2.x_throw:
                    arc_toassign.hand_throw = 'left'
                    arc_toassign2.hand_throw = 'right'
                else:
                    arc_toassign.hand_throw = 'right'
                    arc_toassign2.hand_throw = 'left'
                arc_queue.append(arc_toassign)
                arc_queue.append(arc_toassign2)
                if vocal:
                    print(f'sync pair; set arc {arc_toassign.throw_id} '
                          f'to throw from {arc_toassign.hand_throw}')
                    print(f'sync pair; set arc {arc_toassign2.throw_id} '
                          f'to throw from {arc_toassign2.hand_throw}')
            else:
                arc_toassign.hand_throw = 'right' if (
                        arc_assigned.hand_throw == 'left') else 'left'
                arc_queue.append(arc_toassign)
                if vocal:
                    print(f'alternating (from arc {arc_assigned.throw_id}); '
                          f'set arc {arc_toassign.throw_id} to throw from '
                          f'{arc_toassign.hand_throw}')

        # fill in as many catching hands as we can:
        for arc in run:
            if arc.prev is not None:
                arc.prev.hand_catch = arc.hand_throw
                if vocal and arc.hand_throw is not None:
                    print(f'propagating; set arc {arc.prev.throw_id} '
                          f'to catch in {arc.hand_throw}')

    def estimate_ball_count(self, run_dict):
        """
        Use some heuristics to estimate the number of balls in the pattern.
        This can't be done by counting the number of object in the air since
        there is usually at least one in the hands that won't be seen by
        the tracker.
        """
        notes = self.notes
        run = run_dict['throw']
        tps = run_dict['throws per sec']
        height = notes['cm_per_pixel'] * mean(arc.height for arc in run)

        if tps is None:
            # should never happen
            N_round = N_est = 1
        else:
            # estimate using physics, from the height of the pattern
            g = 980.7               # gravity in cm/s^2
            dwell_ratio = 0.63      # assumed fraction of time hand is filled
            N_est = 2 * dwell_ratio + tps * sqrt(8 * height / g)

            same_side_throws = sum(1 for arc in run if
                                   arc.hand_catch == arc.hand_throw)
            total_throws = sum(1 for arc in run if arc.hand_catch is not None
                               and arc.hand_throw is not None)

            if total_throws > 0:
                if same_side_throws > 0.5 * total_throws:
                    # assume a fountain pattern with even number ->
                    # round N to the nearest even number
                    N_round = 2 * int(round(0.5 * N_est))
                else:
                    # assume a cascade pattern with odd number ->
                    # round N to the nearest odd number
                    N_round = 1 + 2 * int(round(0.5 * (N_est - 1)))
            else:
                N_round = int(round(N_est))

        # maximum possible value based on connections between arcs:
        N_max = sum(1 for arc in run if arc.prev is None)

        run_dict['balls'] = N = min(N_round, N_max)
        if self._verbosity >= 2:
            print(f'tps = {tps:.2f} Hz, H = {height:.1f} cm, '
                  f'N_est = {N_est:.2f}, N_round = {N_round}, '
                  f'N_max = {N_max}, N = {N}')

    def analyze_throw_errors(self, run_dict):
        """
        For each arc, calculate and store as attributes the following:
        (1) location error in throw and peak (x,y) positions
        (2) ideal arc, i.e., version of arc object where position errors (1)
            are zero
        (3) list of close arcs the object comes nearest to
        """
        notes = self.notes

        run = run_dict['throw']
        for arc in run:
            arc.error = None
            arc.ideal = None
            arc.close_arcs = None
        if len(run) < 3:
            return

        balls = run_dict['balls']
        PoverW_ideal = 0.32

        sorted_run = sorted(run, key=lambda a: a.throw_id)

        for throw_idx in range(2, len(run)):
            arc = sorted_run[throw_idx]

            # set of other arcs likeliest to collide with this one:
            if balls % 2 == 0:
                window = [-2, 2]
            else:
                window = [-2, -1, 1, 2]
            window = [x for x in window if 0 <= (throw_idx + x) < len(run)]
            if len(window) < 2:
                continue

            N = X = X2 = F = XF = C = 0
            for x in window:
                arc2 = sorted_run[throw_idx + x]
                N += 1
                X += x
                X2 += x * x
                F += arc2.f_throw
                XF += x * arc2.f_throw
                C += arc2.c     # y-coordinate (pixels) at peak

            # print('len(window) = {}'.format(len(window)))

            f_throw_ideal = (X2 * F - X * XF) / (X2 * N - X * X)
            c_ideal = C / N
            e_ideal = arc.e




            """
            if sum(1 for t in run if t.hand_catch == 'left') == 0:
                continue
            if sum(1 for t in run if t.hand_catch == 'right') == 0:
                continue

            # find the target width and height of the pattern
            catch_left_avg = mean(t.x_catch for t in run
                                             if t.hand_catch == 'left')
            catch_right_avg = mean(t.x_catch for t in run
                                              if t.hand_catch == 'right')
            width = catch_left_avg - catch_right_avg
            height = mean(t.height for t in run)

            if balls == 9:
                ideal_x_throw = (0.5 * width) * 0.55
            else:
                ideal_x_throw = (0.5 * width) * 0.5

            for arc in run:
                # make an 'ideal' arc with the same throw time as arc,
                # but perfect height, throw location, and catch location

                # ideal catch time is calculated using weighted average
                # of catch times for previous catches into the same hand
                prev_arcs = [t for t in run if t.f_catch < arc.f_catch
                             and t.hand_catch == arc.hand_catch]
                if len(prev_arcs) < 2:
                    # need at least two prior catches for least squares
                    # fitting
                    continue

                ideal_arc = Ballarc()
                ideal_arc.f_throw = arc.f_throw
                ideal_arc.e = arc.e
                ideal_arc.x_origin = arc.x_origin
                ideal_arc.y_origin = arc.y_origin

                prev_arcs = sorted(prev_arcs, key=lambda x: x.f_catch,
                                   reverse=True)
                num_to_fit = max(2, floor((balls - 1) / 2))
                A = IA = I2A = AY = IAY = 0.0
                for i, prev_arc in enumerate(prev_arcs[:num_to_fit], start=1):
                    alpha = 1.0 / 2**i
                    A += alpha
                    IA += i * alpha
                    I2A += i * i * alpha
                    AY += alpha * prev_arc.f_catch
                    IAY += i * alpha * prev_arc.f_catch
                ideal_arc.f_catch = (I2A * AY - IA * IAY) / (I2A * A - IA * IA)

                df = 0.5 * (ideal_arc.f_catch - ideal_arc.f_throw)
                ideal_arc.f_peak = ideal_arc.f_throw + df
                height = ideal_arc.e * df**2
                ideal_arc.c = ideal_arc.y_origin - height

                ideal_arc.x_throw = (ideal_x_throw if arc.hand_throw == 'left'
                                     else -ideal_x_throw)
                ideal_arc.x_catch = ((0.5 * width) if arc.hand_catch == 'left'
                                     else -(0.5 * width))

                ideal_arc.a = ideal_arc.x_origin + 0.5 * (ideal_arc.x_catch +
                                                          ideal_arc.x_throw)
                ideal_arc.b = ((ideal_arc.x_catch - ideal_arc.x_throw) /
                               (2.0 * df))

                arc.ideal = ideal_arc
            """

            for arc in run:
                close_arcs = [(a, arc.closest_approach_frame(a, notes))
                              for a in run if a is not arc and
                              a.f_throw < arc.f_catch and
                              a.f_catch > arc.f_throw]
                close_arcs = [(a.throw_id, f,
                               arc.get_distance_from_arc(a, f, notes))
                              for a, f in close_arcs]
                close_arcs = sorted(close_arcs, key=lambda x: x[2])

                arc.close_arcs = close_arcs

    # --- Non-member functions ------------------------------------------------

    def defaultScannerParams():
        """
        Returns a dictionary with constants that configure Hawkeye's video
        scanner. Optionally you can pass a dictionary of this type to the
        HEVideoScanner initializer as 'params'. In most cases the defaults
        should work pretty well.

        The 'high res' values apply when the frame height is greater than or
        equal to 480 pixels.
        """
        params = {
            # duration (in seconds) over which body positions are averaged
            'body_averaging_time_window_secs': 0.2,

            # area (square pixels) of smallest blobs detected
            'min_blob_area_high_res': 7.0,
            'min_blob_area_low_res': 1.0,

            # area (square pixels) of largest blobs detected
            'max_blob_area_high_res': 1000.0,
            'max_blob_area_low_res': 150.0,

            # maximum height of the frame in the juggling plane, in centimeters
            'max_frame_height_cm': 1000,

            # default height of the frame in the juggling plane, in centimeters
            'default_frame_height_cm': 300,

            # assumed maximum frames per second
            'max_fps': 60,

            # default frames per second
            'default_fps': 30,

            # assumed uncertainty in measured locations, in pixels^2
            'sigmasq': 15.0,

            # when building initial arcs from data, largest allowed gap (in
            # frames) between tags for a given arc
            'max_frame_gap_in_arc': 2,

            # closeness to arc to associate a tag with an arc
            'max_distance_pixels_480': 5,

            # how close (fractionally) an arc's acceleration must be to
            # calculated value of g, to be accepted
            'g_window': 0.25,

            # how close (fractionally) a tag's radius needs to be to the median
            # tag radius of an arc, to be allowed to attach to that arc
            'radius_window_high_res': 0.65,      # was 0.3
            'radius_window_low_res': 0.75,

            # minimum number of tags needed for an arc to be considered valid
            'min_tags_per_arc_high_fps': 10,    # was 6 JKB
            'min_tags_per_arc_low_fps': 5,

            # number of tags needed to start curve fitting arc to the data
            'min_tags_to_curve_fit': 4
        }
        return params

    def read_notes(filename):
        """
        Read in the notes data structure from a pickle file.

        Args:
            filename(string):
                filename to read
        Returns:
            notes(dict):
                record of all raw detection events
        """
        with open(filename, 'rb') as handle:
            notes = pickle.load(handle)
        return notes

    def write_notes(notes, filename):
        """
        Write the notes data structure to a pickle file.

        Args:
            notes(dict):
                record of all raw detection events
            filename(string):
                filename to write
        Returns:
            None
        """
        _filepath = os.path.abspath(filename)
        _dirname = os.path.dirname(_filepath)
        if not os.path.exists(_dirname):
            os.makedirs(_dirname)

        with open(_filepath, 'wb') as handle:
            pickle.dump(notes, handle, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------------------------------------------------------


def play_video(filename, notes=None, outfilename=None, startframe=0,
               keywait=False):
    """
    This is not part of the scanner per se but is helpful for testing and
    debugging. It plays a video using OpenCV, including overlays based on
    data in the optional 'notes' dictionary. If 'outfilename' is specified
    then it will write the annotated video to a file on disk.

    Keyboard 'q' quits, 'i' toggles arc number labels.
    """
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print('Error opening video stream or file')
        return
    # cap.set(1, startframe)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if outfilename is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(outfilename, fourcc, fps, (framewidth,
                                                         frameheight))

    tags = dict()
    body = dict()
    arcs = []
    if notes is not None:
        if 'meas' in notes:
            tags = notes['meas']
        if 'body' in notes:
            body = notes['body']
        if 'arcs' in notes:
            arcs = notes['arcs']

    cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # framenum = framereads = startframe
    framenum = framereads = 0
    show_arc_id = False

    while cap.isOpened():
        ret, frame = cap.read()
        framereads += 1

        if not ret:
            print('VideoCapture.read() returned False '
                  'on frame read {}'.format(framereads))
            if framereads > framecount:
                break
            continue

        if framenum >= startframe:
            if framenum in body:
                x, y, w, h, _ = notes['body'][framenum]
                x = int(round(x))
                y = int(round(y))
                w = int(round(w))
                h = int(round(h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if framenum in tags:
                for tag in tags[framenum]:
                    color = ((0, 255, 0) if tag.arc is not None
                             else (0, 0, 255))
                    cv2.circle(frame, (int(round(tag.x)), int(round(tag.y))),
                               int(round(tag.radius)), color, 1)

            for arc in arcs:
                if notes['version'] < 3:
                    start, end = arc.get_frame_range(notes)
                else:
                    start, end = arc.get_tag_range()
                if start <= framenum <= end:
                    x, y = arc.get_position(framenum, notes)
                    x = int(round(x))
                    y = int(round(y))
                    if (notes is not None and len(arc.tags) <
                            notes['scanner_params']['min_tags_per_arc']):
                        temp, _ = arc.get_tag_range()
                        print('error, arc {} at {} has only {} tags'.format(
                            arc.id_, temp, len(arc.tags)))

                    arc_has_tag = any(
                            arc.get_distance_from_tag(tag, notes) <
                            notes['scanner_params']['max_distance_pixels']
                            for tag in arc.tags if tag.frame == framenum)

                    color = (0, 255, 0) if arc_has_tag else (0, 0, 255)
                    cv2.circle(frame, (x, y), 2, color, -1)
                    if show_arc_id:
                        cv2.rectangle(frame, (x+10, y+5),
                                      (x+40, y-4), (0, 0, 0), -1)
                        cv2.putText(frame, format(arc.id_, ' 4d'),
                                    (x+13, y+4),
                                    font, 0.6, (255, 255, 255), 1,
                                    cv2.LINE_AA)

            cv2.rectangle(frame, (3, 3), (80, 27), (0, 0, 0), -1)
            cv2.putText(frame, format(framenum, ' 7d'), (10, 20), font, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            (h, w) = frame.shape[:2]
            r = 640 / float(h)
            dim = (int(w * r), 640)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow(filename, frame)

            if outfilename is not None:
                out.write(frame)

            stop_playback = False
            while not stop_playback:
                keycode = cv2.waitKey(1) & 0xFF
                if keycode == ord('q'):     # Q on keyboard exits
                    stop_playback = True
                    break
                elif keycode == ord('i'):
                    show_arc_id = not show_arc_id
                    break
                if not keywait or keycode != 0xFF:
                    break

            if stop_playback:
                break

        framenum += 1

    cap.release()
    if outfilename is not None:
        out.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------

"""
This entry point isn't used by the Hawkeye application but is useful for
testing and debugging the video scanner.
"""

if __name__ == '__main__':

    # _filename = 'movies/juggling_test_5.mov'
    # _filename = 'movies/TBTB3_9balls.mov'
    # _filename = 'movies/TBTB3_9balls_temp.mp4'
    # _filename = 'movies/GOPR0463.MP4'
    # _filename = 'movies/GOPR0466.MP4'
    # _filename = 'movies/GOPR0467.MP4'
    _filename = 'movies/GOPR0493.MP4'
    # _filename = 'movies/8ballsLonger.mov'
    # _filename = 'movies/vert_test.mp4'
    # _scanvideo = 'movies/__Hawkeye__/GOPR0531_640x480.mp4'
    _scanvideo = 'movies/__Hawkeye__/GOPR0493_640x480.mp4'

    watch_video = False

    # print(cv2.getBuildInformation())

    if watch_video:
        notes_version = 2

        if 1 <= notes_version <= 4:
            _filepath = os.path.abspath(_filename)
            _dirname = os.path.dirname(_filepath)
            _hawkeye_dir = os.path.join(_dirname, '__Hawkeye__')
            _basename = os.path.basename(_filepath)
            _basename_noext = os.path.splitext(_basename)[0]

            if notes_version == 4:
                _basename_notes = _basename_noext + '_notes.pkl'
            else:
                _basename_notes = _basename_noext + '_notes{}.pkl'.format(
                                                            notes_version)
            _filepath_notes = os.path.join(_hawkeye_dir, _basename_notes)

            mynotes = HEVideoScanner.read_notes(_filepath_notes)
        else:
            mynotes = None
        play_video(_filename, notes=mynotes, outfilename=None,
                   startframe=0, keywait=True)
    else:
        startstep = 4
        endstep = 4
        verbosity = 2

        scanner = HEVideoScanner(_filename, scanvideo=_scanvideo)
        scanner.process(steps=(startstep, endstep), readnotes=True,
                        writenotes=True, notesdir='__Hawkeye__',
                        verbosity=verbosity)
