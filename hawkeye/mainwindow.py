# mainwindow.py
#
# Hawkeye application window.
#
# Copyright 2019 Jack Boyce (jboyce@gmail.com)

import os
import sys
import platform
from math import ceil, floor, atan, pi

from PySide2.QtCore import QDir, QSize, Qt, QUrl, QThread, Signal, Slot
from PySide2.QtGui import QFont, QTextCursor, QIcon, QColor, QMovie
from PySide2.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QSlider, QStyle, QVBoxLayout,
                               QWidget, QGraphicsView, QListWidgetItem,
                               QSplitter, QStackedWidget, QPlainTextEdit,
                               QProgressBar, QMainWindow, QAction, QComboBox,
                               QAbstractItemView, QToolButton, QCheckBox,
                               QTableWidget, QTableWidgetItem, QMessageBox,
                               QSpacerItem)
from PySide2.QtMultimedia import QMediaContent, QMediaPlayer, QSoundEffect

from hawkeye.worker import HEWorker
from hawkeye.widgets import (HEVideoView, HEVideoList, HEViewList,
                             HETableViewDelegate)


CURRENT_APP_VERSION = "1.3"
COPYRIGHT_YEAR = "2020"


class HEMainWindow(QMainWindow):
    """
    The main application window.
    """

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle('Hawkeye')

        # currently-selected video and view in UI
        self.current_video_item = None
        self.current_view_item = None

        # fast-stepping playback mode; see HEVideoView.do_playback_control()
        self.step_forward_until = None
        self.step_backward_until = None

        self.prefs = {
            'markers': False,
            'throw_labels': True,
            'parabolas': True,
            'carries': True,
            'ideal_points': True,
            'accuracy_overlay': True,
            'body': False,
            'throw_sounds': False,
            'resolution': 'Actual size',
            'auto_analyze': True
        }

        # Media player on Mac OS bogs down on videos at 1080p resolution,
        # so downsample to 720 as a default.
        if platform.system() == 'Darwin':
            self.prefs['resolution'] = '720'

        self.make_UI()
        self.load_sounds()
        self.grabKeyboard()
        self.start_worker()

    # -------------------------------------------------------------------------
    #  User interface creation
    # -------------------------------------------------------------------------

    def make_UI(self):
        """
        Build all the UI elements in the application window.
        """
        self.make_menus()
        lists_widget = self.make_lists_widget()
        player_widget = self.make_player_widget()
        prefs_widget = self.make_prefs_widget()
        output_widget = self.make_scanner_output_widget()
        data_widget = self.make_data_widget()
        about_widget = self.make_about_widget()

        # assemble window contents
        self.player_stacked_widget = QStackedWidget()
        self.player_stacked_widget.addWidget(player_widget)
        self.player_stacked_widget.addWidget(prefs_widget)
        self.player_stacked_widget.setCurrentIndex(0)

        self.views_stacked_widget = QStackedWidget()
        self.views_stacked_widget.addWidget(self.player_stacked_widget)
        self.views_stacked_widget.addWidget(output_widget)
        self.views_stacked_widget.addWidget(data_widget)
        self.views_stacked_widget.addWidget(about_widget)
        self.views_stacked_widget.setCurrentIndex(3)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(lists_widget)
        splitter.addWidget(self.views_stacked_widget)
        splitter.setCollapsible(1, False)

        self.setCentralWidget(splitter)

    def make_menus(self):
        """
        Make the menu bar for the application.
        """
        open_action = QAction('&Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open movie')
        open_action.triggered.connect(self.open_file)
        exit_action = QAction('&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.exit_call)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(open_action)
        file_menu.addAction(exit_action)

    def make_lists_widget(self):
        """
        Make widget for the left side of the UI, containing the Video and View
        list elements.
        """
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            base_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            base_dir = os.path.dirname(os.path.realpath(__file__))

        lists_layout = QVBoxLayout()
        video_label_layout = QHBoxLayout()
        label = QLabel('Videos:')
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        video_label_layout.addWidget(label)
        video_label_layout.addStretch()
        self.busy_icon = QLabel()
        self.busy_icon.setSizePolicy(QSizePolicy.Preferred,
                                     QSizePolicy.Preferred)
        self.busy_movie = QMovie(os.path.join(base_dir,
                                              'resources/busy_icon.gif'))
        self.busy_icon.setMovie(self.busy_movie)
        video_label_layout.addWidget(self.busy_icon)
        self.busy_icon.hide()
        lists_layout.addLayout(video_label_layout)
        self.video_list = HEVideoList(self)
        self.video_list.itemSelectionChanged.connect(self.on_video_selected)
        self.video_list.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Expanding)
        lists_layout.addWidget(self.video_list)
        label = QLabel('View:')
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lists_layout.addWidget(label)
        self.view_list = HEViewList()
        self.view_list.itemSelectionChanged.connect(self.on_view_selected)
        self.view_list.setSizePolicy(QSizePolicy.Preferred,
                                     QSizePolicy.Expanding)
        lists_layout.addWidget(self.view_list)
        lists_widget = QWidget()
        lists_widget.setLayout(lists_layout)
        return lists_widget

    def make_player_widget(self):
        """
        Make the player widget is the area on the right, with the video player
        and control bar below it.
        """
        self.view = HEVideoView(self)
        self.view.setDragMode(QGraphicsView.NoDrag)

        self.zoomin_button = QPushButton('& + ', self)
        self.zoomin_button.setEnabled(False)
        self.zoomin_button.clicked.connect(self.zoom_in)

        self.zoomout_button = QPushButton('& - ', self)
        # self.zoomout_button = QPushButton('&\u2014', self)
        self.zoomout_button.setEnabled(False)
        self.zoomout_button.clicked.connect(self.zoom_out)

        self.play_button = QPushButton()
        self.play_button.setEnabled(False)
        self.play_button.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        # self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.valueChanged.connect(self.set_position)

        rates = ['0.1', '0.25', '0.5', '1.0', '1.5', '2.0']
        self.rate_combo = QComboBox()
        for index, rate in enumerate(rates):
            self.rate_combo.insertItem(index, rate)
        self.rate_combo.setEditable(False)
        self.rate_combo.currentIndexChanged.connect(self.on_rate_change)
        self.rate_combo.setCurrentIndex(rates.index('0.5'))

        self.back_button = QPushButton()
        self.back_button.setEnabled(False)
        self.back_button.setIcon(self.style().standardIcon(
                QStyle.SP_MediaSkipBackward))
        self.back_button.clicked.connect(self.step_backward)

        self.forward_button = QPushButton()
        self.forward_button.setEnabled(False)
        self.forward_button.setIcon(self.style().standardIcon(
                QStyle.SP_MediaSkipForward))
        self.forward_button.clicked.connect(self.step_forward)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.zoomin_button)
        controls_layout.addWidget(self.zoomout_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.position_slider)
        controls_layout.addWidget(self.rate_combo)
        controls_layout.addWidget(self.back_button)
        controls_layout.addWidget(self.forward_button)

        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            base_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            base_dir = os.path.dirname(os.path.realpath(__file__))

        errorbar_layout = QHBoxLayout()
        errorbar_layout.setContentsMargins(0, 0, 7, 0)
        self.error_label = QLabel()
        self.error_label.setSizePolicy(QSizePolicy.Preferred,
                                       QSizePolicy.Maximum)
        about_button = QToolButton()
        about_button.setIcon(QIcon(
                os.path.join(base_dir, 'resources/about_icon.png')))
        about_button.setIconSize(QSize(20, 20))
        about_button.setStyleSheet('border: none;')
        about_button.clicked.connect(self.show_about)
        prefs_button = QToolButton()
        prefs_button.setIcon(QIcon(
                os.path.join(base_dir, 'resources/preferences_icon.png')))
        prefs_button.setIconSize(QSize(20, 20))
        prefs_button.setStyleSheet('border: none;')
        prefs_button.clicked.connect(self.show_prefs)
        errorbar_layout.addWidget(self.error_label)
        errorbar_layout.addWidget(about_button)
        errorbar_layout.addWidget(prefs_button)

        player_layout = QVBoxLayout()
        # layout_right_player.setContentsMargins(0, 10, 0, 0)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        player_layout.addWidget(self.view)
        player_layout.addLayout(controls_layout)
        player_layout.addLayout(errorbar_layout)
        player_widget = QWidget()
        player_widget.setLayout(player_layout)
        return player_widget

    def make_prefs_widget(self):
        """
        Make the prefs panel that is shown when the settings icon is clicked.
        """
        self.prefs_markers = QCheckBox('Ball detections')
        self.prefs_throwlabels = QCheckBox('Throw number labels')
        self.prefs_parabolas = QCheckBox('Ball arcs')
        self.prefs_carries = QCheckBox('Hand carry distances')
        self.prefs_ideal_points = QCheckBox(
                    'Ideal throw and catch points')
        self.prefs_accuracy_overlay = QCheckBox(
                    'Throwing accuracy overlay')
        self.prefs_body = QCheckBox('Body position')
        self.prefs_sounds = QCheckBox('Throw sounds')

        resolution_layout = QHBoxLayout()
        resolution_layout.setAlignment(Qt.AlignLeft)
        resolution_layout.addWidget(QLabel(
                    'Maximum video display resolution (vertical pixels):'))
        resolutions = ['480', '720', '1080', 'Actual size']
        self.prefs_resolution = QComboBox()
        for index, rate in enumerate(resolutions):
            self.prefs_resolution.insertItem(index, rate)
        self.prefs_resolution.setEditable(False)
        resolution_layout.addWidget(self.prefs_resolution)

        self.prefs_autoanalyze = QCheckBox(
            'Automatically analyze added videos for juggling')

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(10, 0, 30, 0)
        controls_layout.setAlignment(Qt.AlignVCenter)
        controls_layout.addWidget(QLabel('Video playback options:'))
        controls_layout.addWidget(self.prefs_markers)
        controls_layout.addWidget(self.prefs_throwlabels)
        controls_layout.addWidget(self.prefs_parabolas)
        controls_layout.addWidget(self.prefs_carries)
        controls_layout.addWidget(self.prefs_ideal_points)
        controls_layout.addWidget(self.prefs_accuracy_overlay)
        controls_layout.addWidget(self.prefs_body)
        controls_layout.addWidget(self.prefs_sounds)
        controls_layout.addLayout(resolution_layout)
        controls_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding,
                                            QSizePolicy.Minimum))
        controls_layout.addWidget(self.prefs_autoanalyze)
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Expanding)

        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.cancel_prefs)
        accept_button = QPushButton('Accept')
        accept_button.clicked.connect(self.set_prefs)
        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(accept_button)
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)
        buttons_widget.setSizePolicy(QSizePolicy.Preferred,
                                     QSizePolicy.Preferred)

        prefs_layout = QVBoxLayout()
        prefs_layout.addWidget(controls_widget)
        prefs_layout.addWidget(buttons_widget)
        prefs_widget = QWidget()
        prefs_widget.setLayout(prefs_layout)
        return prefs_widget

    def make_scanner_output_widget(self):
        """
        Make a big text box and progress bar widget for showing output during
        the scanning process.
        """
        self.output_widget = QPlainTextEdit()
        self.output_widget.setReadOnly(True)
        self.output_widget.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = self.output_widget.font()
        font.setFamily('Courier')
        font.setStyleHint(QFont.Monospace)
        self.output_widget.setFont(font)
        self.output_widget.setSizePolicy(QSizePolicy.Expanding,
                                         QSizePolicy.Expanding)
        self.progressbar = QProgressBar()
        self.progressbar.setRange(0, 0)

        output_layout = QVBoxLayout()
        label = QLabel('Scanner output:')
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        output_layout.addWidget(label)
        output_layout.addWidget(self.output_widget)
        output_layout.addWidget(self.progressbar)
        output_widget = QWidget()
        output_widget.setLayout(output_layout)
        return output_widget

    def make_data_widget(self):
        """
        Make a table widget showing relevant data for each throw detected in
        the video. This data can be exported to a CSV file.
        """
        data_layout = QVBoxLayout()
        self.data_widget = QTableWidget()
        self.data_widget.verticalHeader().setVisible(False)
        self.data_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_widget.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        self.data_widget.setShowGrid(False)
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignRight)
        export_button = QPushButton('Export as CSV')
        export_button.clicked.connect(self.export_data)
        button_layout.addWidget(export_button)

        data_layout.addWidget(self.data_widget)
        data_layout.addLayout(button_layout)
        data_widget = QWidget()
        data_widget.setLayout(data_layout)
        return data_widget

    def make_about_widget(self):
        """
        Make the info panel widget that is shown on startup, or when the user
        clicks the '?' icon in the lower right.
        """
        msg = ('<p><h2>Hawkeye Juggling Video Analyzer</h2></p>'
               '<p>Version ' + CURRENT_APP_VERSION + '</p>'
               '<p>Copyright &copy; ' + COPYRIGHT_YEAR
               + ' Jack Boyce (jboyce@gmail.com)</p>'
               '<p>&nbsp;</p>'
               '<p>This software processes video files and tracks '
               'objects moving through the air in parabolic trajectories. '
               'Currently it detects balls only. For best results use video '
               'with a stationary camera, four or more objects, a high frame '
               'rate (60+ fps), minimum motion blur, and good brightness '
               'separation between the balls and background.</p>'
               '<p>Drop a video file onto the <b>Videos:</b> box to begin!</p>'
               '<p>Useful keyboard shortcuts in video view:</p>'
               '<ul>'
               '<li>space: toggle play/pause</li>'
               '<li>left/right arrows: step backward/forward by one frame '
               '(hold to continue cueing)</li>'
               '<li>z, x: step backward/forward by one throw</li>'
               '<li>a, s: step backward/forward by one run</li>'
               '<li>down arrow: toggle all overlays/sounds</li>'
               '<li>up arrow: toggle accuracy-related overlays</li>'
               '<li>k: save video clip of current run</li>'
               '<li>p: process video to analyze for juggling</li>'
               '</ul>'
               '<p>&nbsp;</p>'
               '<p><small>All portions of this software written by Jack Boyce '
               'are provided under the MIT License. Other software '
               'distributed as part of Hawkeye is done so under the terms of '
               'their respective non-commercial licenses: '
               'OpenCV version 3.4.4 (3-clause BSD License), '
               'YOLOv2-tiny (MIT License), '
               'Qt version 5.6.2 (LGPL v2.1), '
               'PySide2 version 5.6.0 (LGPL v2.1), '
               'FFmpeg/FFprobe version 4.1.1 (LGPL v2.1).</small></p>'
               )

        text_widget = QLabel(msg)
        text_widget.setWordWrap(True)
        msg_layout = QHBoxLayout()
        msg_layout.setContentsMargins(10, 0, 30, 0)
        msg_layout.addWidget(text_widget)
        msg_widget = QWidget()
        msg_widget.setLayout(msg_layout)
        msg_widget.setSizePolicy(QSizePolicy.Preferred,
                                 QSizePolicy.Expanding)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        ok_button = QPushButton('OK')
        ok_button.clicked.connect(self.cancel_about)
        button_layout.addWidget(ok_button)
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        button_widget.setSizePolicy(QSizePolicy.Preferred,
                                    QSizePolicy.Preferred)

        about_layout = QVBoxLayout()
        about_layout.addWidget(msg_widget)
        about_layout.addWidget(button_widget)
        about_widget = QWidget()
        about_widget.setLayout(about_layout)
        return about_widget

    # -------------------------------------------------------------------------
    #  Load sound resources
    # -------------------------------------------------------------------------

    def load_sounds(self):
        """
        Load sound file that is played during a throw.
        """
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            base_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            base_dir = os.path.dirname(os.path.realpath(__file__))

        throw_wav_file = os.path.join(base_dir,
                                      'resources/billiard-balls.wav')
        self.throwsound = QSoundEffect()
        self.throwsound.setSource(QUrl.fromLocalFile(throw_wav_file))
        self.throwsound.setVolume(0.5)

    # -------------------------------------------------------------------------
    #  Worker thread
    # -------------------------------------------------------------------------

    # signal that informs worker of new display preferences
    sig_new_prefs = Signal(dict)

    # signal that informs worker to transcode a video, and optionally
    # analyze juggling
    sig_process_video = Signal(str, bool)

    # signal that informs worker to analyze a video for juggling content
    sig_analyze_juggling = Signal(str, dict)

    # signal that informs worker to extract a clip from a video
    sig_extract_clip = Signal(str, dict, int)

    def start_worker(self):
        """
        Start up worker object on a separate thread, to handle the lengthy
        processing operations without blocking the UI.

        For data consistency all communication with the worker is done via Qt
        signals and slots.

        There are four task types we can send to the worker:
            1) notify of new app preferences
            2) transcode a video to create a display version
            3) analyze a video for juggling content
            4) extract a clip of a given run and save as a separate video
        When a new video is added, we signal the worker thread to do 2) and
        (depending on prefs) may also signal 3).

        Communication protocol with worker:
            - Worker signals on_worker_output() with any textual output
            - Worker signals on_worker_step() to indicate degree of completion
            - Worker signals on_worker_error() in the event of any error
              condition
            - Worker signals on_worker_***_done() when the task is completed,
              including a return code to indicate success or failure
        """
        self._worker = worker = HEWorker(self.app)
        self._thread = thread = QThread()
        worker.moveToThread(thread)

        # signals from UI thread to worker
        self.sig_new_prefs.connect(worker.on_new_prefs)
        self.sig_process_video.connect(worker.on_process_video)
        self.sig_analyze_juggling.connect(worker.on_analyze_juggling)
        self.sig_extract_clip.connect(worker.on_extract_clip)

        # signals from worker back to UI thread
        worker.sig_output.connect(self.on_worker_output)
        worker.sig_progress.connect(self.on_worker_step)
        worker.sig_error.connect(self.on_worker_error)
        worker.sig_video_done.connect(self.on_worker_displayvid_done)
        worker.sig_analyze_done.connect(self.on_worker_analyze_done)
        worker.sig_clipping_done.connect(self.on_worker_clipping_done)

        # have worker thread clean up worker object on close; otherwise we
        # can get a "QObject::~QObject: Timers cannot be stopped from another
        # thread" console error
        thread.finished.connect(worker.deleteLater)

        thread.start(QThread.LowPriority)   # start thread's event loop

        self.sig_new_prefs.emit(self.prefs)
        self.worker_processing_queue_length = 0
        self.worker_clipping_queue_length = 0
        self.set_worker_busy_icon()

    @Slot(str, str)
    def on_worker_output(self, file_id: str, output: str):
        """
        Receive text output from worker during processing.
        """
        for i in range(0, self.video_list.count()):
            item = self.video_list.item(i)
            if item is None:
                continue
            if item.vc.filepath == file_id:
                # continue to take output even if processing is done
                item.vc.output += output
                if item is self.current_video_item:
                    self.output_widget.moveCursor(QTextCursor.End)
                    self.output_widget.insertPlainText(output)
                    self.output_widget.moveCursor(QTextCursor.End)
                break

    @Slot(str, int, int)
    def on_worker_step(self, file_id: str, step: int, stepstotal: int):
        """
        Receive progress update from worker while processing a video.
        """
        for i in range(0, self.video_list.count()):
            item = self.video_list.item(i)
            if item is None:
                continue
            if item.vc.filepath == file_id:
                item.vc.processing_step = step
                item.vc.processing_steps_total = stepstotal
                if item is self.current_video_item:
                    self.progressbar.setValue(step)
                    self.progressbar.setMaximum(stepstotal)
                break

    @Slot(str, str)
    def on_worker_error(self, file_id: str, errormsg: str):
        """
        Receive notification from worker when there is a processing error.

        When file_id is an empty string it signifies a general nonfatal error
        in the worker thread, which we show to the user.
        """
        if file_id != "":
            for i in range(0, self.video_list.count()):
                item = self.video_list.item(i)
                if item is None:
                    continue
                if item.vc.filepath == file_id:
                    if item.vc.error is None:
                        item.vc.error = errormsg
                    else:
                        item.vc.error += "\n" + errormsg
                    break
        else:
            QMessageBox.warning(self, 'Oops, we encountered a problem.',
                                errormsg, QMessageBox.Ok, QMessageBox.Ok)

    @Slot(str, dict, int, dict, bool)
    def on_worker_displayvid_done(self, file_id: str, fileinfo: dict,
                                  resolution: int, notes: dict, success: bool):
        """
        Receive notification from worker when the transcoded display version
        of a video is completed.

        This will always come before video analysis completion below.
        """
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if item is None:
                continue
            if item.vc.filepath == file_id:
                if not item.vc.analyze_on_add:
                    # no subsequent analysis step queued up
                    item.vc.doneprocessing = True
                    self.worker_processing_queue_length -= 1
                    self.set_worker_busy_icon()
                    item.setForeground(item.foreground)

                if success:
                    item.vc.videopath = fileinfo['displayvid_path']
                    item.vc.videoresolution = resolution
                    item.vc.csvpath = fileinfo['csvfile_path']
                    item.vc.notes = notes
                    # The following defines the range of frames accessible in
                    # the viewer: Frame number can go from 0 to frames-1
                    # inclusive. We shave some frames off the end because we
                    # don't want the player to reach the actual end of media,
                    # and there's unlikely to be anything interesting there.
                    item.vc.frames = notes['frame_count_estimate'] - 4
                    item.vc.duration = int(item.vc.frames * 1000
                                           / notes['fps'])
                    item.vc.position = 0
                    item.vc.has_played = False

                    # Define a function that we will use to draw overlays on
                    # the video. It maps an (x, y) pixel position in the
                    # original video to the equivalent position in the display
                    # video. See HEVideoView.draw_overlays().
                    if item.vc.videoresolution == 0:
                        def map_to_display_video(x, y):
                            return x, y
                    else:
                        orig_height = notes['frame_height']
                        orig_width = notes['frame_width']
                        display_height = item.vc.videoresolution
                        display_width = round(orig_width * display_height
                                              / orig_height)
                        if display_width % 2 == 1:
                            display_width += 1

                        def map_to_display_video(x, y):
                            new_x = x * display_width / orig_width
                            new_y = y * display_height / orig_height
                            return new_x, new_y
                    item.vc.map = map_to_display_video

                    # load the video into the player
                    item.vc.player.pause()
                    item.vc.player.setMedia(QMediaContent(
                            QUrl.fromLocalFile(item.vc.videopath)))
                    item.vc.player.setPosition(0)
                    item.vc.player.pause()

                    if item is self.current_video_item:
                        self.play_button.setEnabled(True)
                        # block signals so we don't trigger set_position()
                        prev = self.position_slider.blockSignals(True)
                        self.position_slider.setRange(0, item.vc.duration)
                        self.position_slider.setValue(0)
                        self.position_slider.blockSignals(prev)
                        self.build_view_list(item)
                        if not item.vc.analyze_on_add:
                            # switch to movie view
                            self.views_stacked_widget.setCurrentIndex(0)
                elif not item.vc.analyze_on_add:
                    if item is self.current_video_item:
                        self.progressbar.hide()
                        self.build_view_list(item)
                    self.show_worker_errors(item)
                break

    @Slot(str, dict, bool)
    def on_worker_analyze_done(self, file_id: str, notes: dict, success: bool):
        """
        Receive notification from worker when analyzing a video for juggling
        content is completed.

        This will always come after the display video completion.
        """
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if item is None:
                continue
            if item.vc.filepath == file_id:
                item.vc.doneprocessing = True
                self.worker_processing_queue_length -= 1
                self.set_worker_busy_icon()

                if success:
                    item.vc.notes = notes
                    # recalculate duration in case actual frame count is
                    # different from estimated frame count
                    item.vc.frames = notes['frame_count'] - 4
                    item.vc.duration = int(item.vc.frames * 1000
                                           / notes['fps'])

                    if item.vc.position == 0 and notes['runs'] > 0:
                        # if we haven't started playing the video, set its
                        # position to just before the first run
                        startframe, _ = notes['run'][0]['frame range']
                        # one second before run's starting frame
                        startframe = max(0, floor(startframe - notes['fps']))
                        item.vc.position = position_for_framenum(item.vc,
                                                                 startframe)
                        item.vc.player.setPosition(item.vc.position)
                        item.vc.player.pause()      # necessary on Mac?

                    if item is self.current_video_item:
                        self.progressbar.hide()
                        # block signals so we don't trigger set_position()
                        prev = self.position_slider.blockSignals(True)
                        self.position_slider.setRange(0, item.vc.duration)
                        self.position_slider.setValue(item.vc.position)
                        self.position_slider.blockSignals(prev)
                        self.build_view_list(item)
                        # switch to movie view
                        self.views_stacked_widget.setCurrentIndex(0)

                    item.setForeground(item.foreground)
                else:
                    if item is self.current_video_item:
                        self.progressbar.hide()
                        self.build_view_list(item)
                    self.show_worker_errors(item)
                break

    @Slot(str, bool)
    def on_worker_clipping_done(self, file_id: str, success: bool):
        """
        Receive notification from worker when clipping a run from a video is
        completed.
        """
        self.worker_clipping_queue_length -= 1
        self.set_worker_busy_icon()

        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if item is None:
                continue
            if item.vc.filepath == file_id:
                if success:
                    item.setForeground(item.foreground)
                else:
                    self.show_worker_errors(item)
                break

    def show_worker_errors(self, videoitem):
        """
        Show in the scanner output window any error strings that were returned
        from processing.
        """
        videoitem.setForeground(Qt.red)

        output = '\n\n##### An error occurred while processing video #####\n'
        if videoitem.vc.error is not None:
            output += videoitem.vc.error
        self.on_worker_output(videoitem.vc.filepath, output)

        if videoitem is self.current_video_item:
            # switch to 'scanner output' view
            for i in range(self.view_list.count()):
                viewitem = self.view_list.item(i)
                if viewitem._type == 'output':
                    viewitem.setSelected(True)
                    break

    def is_worker_busy(self):
        """
        Return True when worker is busy, False when idle.
        """
        return (self.worker_processing_queue_length
                + self.worker_clipping_queue_length != 0)

    def set_worker_busy_icon(self):
        """
        Show or hide the "worker busy" icon in the UI based on the state of
        the worker.

        Call this whenever the state of the worker changes.
        """
        if self.is_worker_busy():
            self.busy_icon.show()
            self.busy_movie.start()
        else:
            self.busy_icon.hide()
            self.busy_movie.stop()

    # -------------------------------------------------------------------------
    #  UI interactions
    # -------------------------------------------------------------------------

    @Slot()
    def on_video_selected(self):
        """
        Receive notification of an item selected in the HEVideoList.

        This is called either through user action or programmatically (the
        latter happens when a file is dropped on the HEVideoList).
        """
        items = self.video_list.selectedItems()
        if len(items) == 0:
            return
        item = items[0]
        if item is self.current_video_item:
            return

        # pause video player for current movie, if one is showing
        self.pause_movie()

        self.current_video_item = item

        # set up video viewer
        # attach HEVideoView to this video's media player:
        self.view.setScene(item.vc.graphicsscene)
        item.vc.player.setPlaybackRate(float(
                    self.rate_combo.currentText()))
        self.step_forward_until = None
        self.step_backward_until = None

        # set up UI elements
        self.play_button.setEnabled(item.vc.videopath is not None)
        self.back_button.setEnabled(item.vc.has_played)
        self.forward_button.setEnabled(item.vc.has_played)
        self.error_label.setText('')
        if item.vc.duration is not None:
            # block signals so we don't trigger set_position()
            prev = self.position_slider.blockSignals(True)
            self.position_slider.setRange(0, item.vc.duration)
            self.position_slider.setValue(item.vc.position)
            self.position_slider.blockSignals(prev)

        self.build_view_list(item)

        if item.vc.doneprocessing and item.vc.videopath is not None:
            # switch to video view
            self.views_stacked_widget.setCurrentIndex(0)
        else:
            # switch to 'scanner output' view
            for i in range(self.view_list.count()):
                viewitem = self.view_list.item(i)
                if viewitem._type == 'output':
                    viewitem.setSelected(True)
                    break

    def build_view_list(self, videoitem):
        """
        Construct the set of views to make available for a given video. This
        populates the View list in the UI.
        """
        notes = videoitem.vc.notes
        self.view_list.clear()

        have_video = videoitem.vc.videopath is not None
        did_analysis = (notes is not None and 'step' in notes
                        and notes['step'] >= 6)
        have_runs = (notes is not None and 'runs' in notes
                     and notes['runs'] > 0)

        if did_analysis:
            if have_runs:
                headeritem = QListWidgetItem('')
                headeritem._type = 'blank'
                headeritem.setFlags(headeritem.flags()
                                    & ~Qt.ItemIsSelectable)
                header = QLabel('Run #, Balls, Throws')
                self.view_list.addItem(headeritem)
                self.view_list.setItemWidget(headeritem, header)

                for i in range(notes['runs']):
                    run_dict = notes['run'][i]
                    runitem = QListWidgetItem('')
                    runitem._type = 'run'
                    runitem._runindex = i
                    startframe, _ = run_dict['frame range']
                    # start one second before run's starting frame
                    startframe = max(0, floor(startframe - notes['fps']))
                    runitem._startframe = startframe
                    if (i + 1) < notes['runs']:
                        endframe, _ = notes['run'][i + 1]['frame range']
                        endframe = max(0, floor(endframe - notes['fps']))
                    else:
                        endframe = notes['frame_count']
                    runitem._endframe = endframe
                    runitem.setFlags(headeritem.flags()
                                     | Qt.ItemIsSelectable)
                    run = QLabel('{}, {}, {}'. format(i + 1,
                                                      run_dict['balls'],
                                                      run_dict['throws']))
                    self.view_list.addItem(runitem)
                    self.view_list.setItemWidget(runitem, run)
            else:
                headeritem = QListWidgetItem('')
                headeritem._type = 'blank'
                headeritem.setFlags(headeritem.flags()
                                    & ~Qt.ItemIsSelectable)
                header = QLabel('(no runs detected)')
                self.view_list.addItem(headeritem)
                self.view_list.setItemWidget(headeritem, header)

            headeritem = QListWidgetItem('')
            headeritem._type = 'blank'
            headeritem.setFlags(headeritem.flags() & ~Qt.ItemIsSelectable)
            self.view_list.addItem(headeritem)

        if have_video:
            headeritem = QListWidgetItem('')
            headeritem._type = 'video'
            headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
            header = QLabel('Video')
            self.view_list.addItem(headeritem)
            self.view_list.setItemWidget(headeritem, header)

        if have_runs:
            headeritem = QListWidgetItem('')
            headeritem._type = 'data'
            headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
            header = QLabel('Run data')
            self.view_list.addItem(headeritem)
            self.view_list.setItemWidget(headeritem, header)

        headeritem = QListWidgetItem('')
        headeritem._type = 'output'
        headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
        header = QLabel('Scanner output')
        self.view_list.addItem(headeritem)
        self.view_list.setItemWidget(headeritem, header)

        if not videoitem.vc.doneprocessing:
            headeritem = QListWidgetItem('')
            headeritem._type = 'blank'
            headeritem.setFlags(headeritem.flags() & ~Qt.ItemIsSelectable)
            header = QLabel('(processing...)')
            header.setStyleSheet('QLabel { color : gray }')
            self.view_list.addItem(headeritem)
            self.view_list.setItemWidget(headeritem, header)

    @Slot()
    def on_view_selected(self):
        """
        Called when an item is selected in the View list, either through user
        action or programmatically.
        """
        items = self.view_list.selectedItems()
        if len(items) == 0:
            return
        item = items[0]
        self.current_view_item = item

        if item._type == 'video':
            self.views_stacked_widget.setCurrentIndex(0)
        elif item._type == 'output':
            self.pause_movie()
            self.output_widget.setPlainText(self.current_video_item.vc.output)
            if self.current_video_item.vc.doneprocessing:
                self.progressbar.hide()
            else:
                # print("calling ensureCursorVisible()")
                self.output_widget.moveCursor(QTextCursor.End)
                self.output_widget.ensureCursorVisible()
                # vsb = self.output_widget.verticalScrollBar()
                # if vsb is not None:
                #    vsb.setValue(vsb.maximum())
                self.progressbar.setValue(
                        self.current_video_item.vc.processing_step)
                self.progressbar.setMaximum(
                        self.current_video_item.vc.processing_steps_total)
                self.progressbar.show()
            self.views_stacked_widget.setCurrentIndex(1)
        elif item._type == 'data':
            notes = self.current_video_item.vc.notes
            if notes is not None:
                self.pause_movie()
                self.fill_data_view(notes)
                self.views_stacked_widget.setCurrentIndex(2)
        elif item._type == 'run':
            self.views_stacked_widget.setCurrentIndex(0)
            self.play_movie()
            self.set_framenum(item._startframe)
        else:
            # shouldn't ever get here
            pass

    def fill_data_view(self, notes: dict):
        """
        Fill in the data view table.
        """
        headers = ['Run #', 'Throw #', 'Hands',
                   'Throw x (cm)', 'Catch x (cm)',
                   'Angle (deg)', 'Height (cm)',
                   'Throw t (s)', 'Catch t (s)',
                   'Torso x (cm)',
                   'Prev throw', 'Next throw']
        alignments = [Qt.AlignHCenter, Qt.AlignHCenter, Qt.AlignHCenter,
                      Qt.AlignRight, Qt.AlignRight,
                      Qt.AlignRight, Qt.AlignRight,
                      Qt.AlignRight, Qt.AlignRight,
                      Qt.AlignRight,
                      Qt.AlignHCenter, Qt.AlignHCenter]

        dw = self.data_widget
        dw.setRowCount(len(notes['arcs']))
        dw.setColumnCount(len(headers))
        for col, name in enumerate(headers):
            item = QTableWidgetItem(name)
            item.setTextAlignment(alignments[col] | Qt.AlignVCenter)
            dw.setHorizontalHeaderItem(col, item)

        row = 0
        boundary_ix = []
        right_throws_color = QColor(230, 230, 230)

        for run_id in range(1, notes['runs'] + 1):
            for throw_id in range(1, notes['run'][run_id-1]['throws'] + 1):
                arc = next((a for a in notes['arcs'] if a.run_id == run_id and
                            a.throw_id == throw_id), None)
                if arc is None:
                    continue

                throw_t = arc.f_throw / notes['fps']
                if throw_id == 1:
                    start_t = throw_t
                throw_t -= start_t
                throw_x, _ = arc.get_position(arc.f_throw, notes)
                throw_x *= notes['cm_per_pixel']
                if arc.hand_throw == 'right':
                    throw_hand = 'R'
                elif arc.hand_throw == 'left':
                    throw_hand = 'L'
                else:
                    throw_hand = '?'
                catch_t = arc.f_catch / notes['fps'] - start_t
                catch_x, _ = arc.get_position(arc.f_catch, notes)
                catch_x *= notes['cm_per_pixel']
                if arc.hand_catch == 'right':
                    catch_hand = 'R'
                elif arc.hand_catch == 'left':
                    catch_hand = 'L'
                else:
                    catch_hand = '?'
                hands = throw_hand + '->' + catch_hand

                throw_vx = arc.b
                throw_vy = -2 * arc.e * (arc.f_throw - arc.f_peak)
                angle = atan(throw_vx / throw_vy) * 180 / pi
                height = 0.125 * 980.7 * (catch_t - throw_t)**2

                try:
                    x, _, w, _, _ = notes['body'][round(arc.f_peak)]
                    body_x = (x + 0.5 * w) * notes['cm_per_pixel']
                    body_x_str = f'{body_x:.1f}'
                    throw_x -= body_x
                    catch_x -= body_x
                except KeyError:
                    body_x_str = '-'
                prev_id = '-' if arc.prev is None else str(arc.prev.throw_id)
                next_id = '-' if arc.next is None else str(arc.next.throw_id)

                dw.setItem(row, 0, QTableWidgetItem(str(arc.run_id)))
                dw.setItem(row, 1, QTableWidgetItem(str(arc.throw_id)))
                dw.setItem(row, 2, QTableWidgetItem(hands))
                dw.setItem(row, 3, QTableWidgetItem(f'{throw_x:.1f}'))
                dw.setItem(row, 4, QTableWidgetItem(f'{catch_x:.1f}'))
                dw.setItem(row, 5, QTableWidgetItem(f'{angle:.2f}'))
                dw.setItem(row, 6, QTableWidgetItem(f'{height:.1f}'))
                dw.setItem(row, 7, QTableWidgetItem(f'{throw_t:.3f}'))
                dw.setItem(row, 8, QTableWidgetItem(f'{catch_t:.3f}'))
                dw.setItem(row, 9, QTableWidgetItem(body_x_str))
                dw.setItem(row, 10, QTableWidgetItem(prev_id))
                dw.setItem(row, 11, QTableWidgetItem(next_id))
                for col, align in enumerate(alignments):
                    item = dw.item(row, col)
                    item.setTextAlignment(align | Qt.AlignVCenter)
                    if throw_hand == 'R':
                        item.setBackground(right_throws_color)
                row += 1
            if run_id < notes['runs']:
                boundary_ix.append(row)

        # item delegate draws lines between runs:
        dw.setItemDelegate(HETableViewDelegate(dw, boundary_ix))
        dw.resizeColumnsToContents()
        dw.resizeRowsToContents()

    @Slot()
    def export_data(self):
        """
        Called when the user clicks the 'Export as CSV' button in data view.
        """
        data = []
        line = []
        dw = self.data_widget
        for col in range(dw.columnCount()):
            line.append(str(dw.horizontalHeaderItem(col).text()))
        data.append(', '.join(line))
        for row in range(dw.rowCount()):
            line = []
            for col in range(dw.columnCount()):
                line.append(str(dw.item(row, col).text()))
            data.append(', '.join(line))
        output = '\n'.join(data)

        csvfile_path = self.current_video_item.vc.csvpath
        filename = QFileDialog.getSaveFileName(self, 'Save CSV File',
                                               csvfile_path)
        if filename != '':
            filepath = os.path.abspath(filename[0])
            with open(filepath, 'w') as f:
                f.write(output)

    @Slot()
    def show_prefs(self):
        """
        Called when the user clicks the preferences icon in the UI.
        """
        self.pause_movie()
        self.prefs_markers.setChecked(self.prefs['markers'])
        self.prefs_throwlabels.setChecked(self.prefs['throw_labels'])
        self.prefs_parabolas.setChecked(self.prefs['parabolas'])
        self.prefs_carries.setChecked(self.prefs['carries'])
        self.prefs_ideal_points.setChecked(self.prefs['ideal_points'])
        self.prefs_accuracy_overlay.setChecked(self.prefs['accuracy_overlay'])
        self.prefs_body.setChecked(self.prefs['body'])
        self.prefs_sounds.setChecked(self.prefs['throw_sounds'])
        self.prefs_resolution.setCurrentText(self.prefs['resolution'])
        self.prefs_autoanalyze.setChecked(self.prefs['auto_analyze'])
        self.player_stacked_widget.setCurrentIndex(1)

    @Slot()
    def set_prefs(self):
        """
        Called when the user clicks "Accept" in the preferences panel.
        """
        old_resolution = self.prefs['resolution']
        self.prefs['markers'] = self.prefs_markers.isChecked()
        self.prefs['throw_labels'] = self.prefs_throwlabels.isChecked()
        self.prefs['parabolas'] = self.prefs_parabolas.isChecked()
        self.prefs['carries'] = self.prefs_carries.isChecked()
        self.prefs['ideal_points'] = self.prefs_ideal_points.isChecked()
        self.prefs['accuracy_overlay'] = \
            self.prefs_accuracy_overlay.isChecked()
        self.prefs['body'] = self.prefs_body.isChecked()
        self.prefs['throw_sounds'] = self.prefs_sounds.isChecked()
        self.prefs['resolution'] = self.prefs_resolution.currentText()
        self.prefs['auto_analyze'] = self.prefs_autoanalyze.isChecked()
        self.sig_new_prefs.emit(self.prefs)
        self.player_stacked_widget.setCurrentIndex(0)

        # If display resolution changed, reprocess all videos through worker
        # to create the needed video files. Do the current video first.
        if self.prefs['resolution'] != old_resolution:
            if self.current_video_item is not None:
                self.current_video_item.vc.doneprocessing = False
                self.sig_process_video.emit(
                        self.current_video_item.vc.filepath, False)
                self.worker_processing_queue_length += 1

                # auto-select 'scanner output' view
                for i in range(self.view_list.count()):
                    viewitem = self.view_list.item(i)
                    if viewitem._type == 'output':
                        viewitem.setSelected(True)
                        break

            for i in range(self.video_list.count()):
                item = self.video_list.item(i)
                if item is not None and item is not self.current_video_item:
                    item.vc.doneprocessing = False
                    self.sig_process_video.emit(item.vc.filepath, False)
                    self.worker_processing_queue_length += 1

            self.set_worker_busy_icon()

    @Slot()
    def cancel_prefs(self):
        """
        Called when the user clicks "Cancel" in the preferences panel.
        """
        self.player_stacked_widget.setCurrentIndex(0)

    @Slot()
    def show_about(self):
        """
        Called when the user clicks the about icon in the UI.
        """
        self.pause_movie()
        self.views_stacked_widget.setCurrentIndex(3)

    @Slot()
    def cancel_about(self):
        """
        Called when the user clicks 'OK' in the about panel.
        """
        self.views_stacked_widget.setCurrentIndex(0)

    @Slot()
    def zoom_in(self):
        """
        Called when the user clicks the zoom in button in the UI.
        """
        self.view.scale(1.2, 1.2)
        self.zoomout_button.setEnabled(True)
        self.view.videosnappedtoframe = False
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

    @Slot()
    def zoom_out(self):
        """
        Called when the user clicks the zoom out button in the UI.
        """
        if not self.view.videosnappedtoframe:
            self.view.scale(1/1.2, 1/1.2)

            # see if the entire frame is now visible; if so then snap to frame
            port_rect = self.view.viewport().rect()
            scene_polygon = self.view.mapToScene(port_rect)
            graphicsvideoitem = self.current_video_item.vc.graphicsvideoitem
            item_polygon = graphicsvideoitem.mapFromScene(scene_polygon)
            item_rect = item_polygon.boundingRect()
            if item_rect.contains(graphicsvideoitem.boundingRect()):
                self.view.fitInView(graphicsvideoitem.boundingRect(),
                                    Qt.KeepAspectRatio)
                self.zoomout_button.setEnabled(False)
                self.view.videosnappedtoframe = True
                self.view.setDragMode(QGraphicsView.NoDrag)

    @Slot()
    def step_backward(self):
        """
        Called when the user clicks the 'back one frame' button in the UI.
        """
        if self.current_video_item.vc.notes is None:
            return
        self.current_video_item.vc.player.pause()
        self.set_framenum(self.framenum() - 1)

    @Slot()
    def step_forward(self):
        """
        Called when the user clicks the 'forward one frame' button in the UI.
        """
        if self.current_video_item.vc.notes is None:
            return
        self.current_video_item.vc.player.pause()
        self.set_framenum(self.framenum() + 1)

    @Slot(int)
    def on_rate_change(self, index: int):
        """
        Called when the user selects a playback rate on the dropdown menu.
        """
        if self.current_video_item is None:
            return
        self.current_video_item.vc.player.setPlaybackRate(float(
                    self.rate_combo.currentText()))

    @Slot()
    def open_file(self):
        """
        Called when the user chooses 'Open Video' from the File menu.
        """
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Video',
                                                  QDir.homePath())
        if filename != '':
            filepath = os.path.abspath(filename)
            lastitem = self.video_list.add_video(filepath)
            if lastitem is not None:
                lastitem.setSelected(True)

    @Slot()
    def toggle_play(self):
        """
        Called when the user clicks the play/pause button in the UI.
        """
        self.step_backward_until = self.step_forward_until = None

        if (self.current_video_item.vc.player.state() ==
                QMediaPlayer.PlayingState):
            self.pause_movie()
        else:
            self.play_movie()

    @Slot(int)
    def set_position(self, position):
        """
        Called when user interacts with the position slider.

        This is NOT called when the position slider changes as a result of
        playing the movie. See HEVideoView.do_playback_control().
        """
        vc = self.current_video_item.vc
        if vc.notes is not None:
            self.set_framenum(framenum_for_position(vc, position))
        else:
            vc.player.setPosition(position)

    @Slot()
    def exit_call(self, close_event=None):
        """
        Called when the user selects Quit in the menu, or clicks the close
        box in the window's corner. This does a clean exit of the application.
        """
        self.pause_movie()
        wants_to_quit = True

        if self.is_worker_busy():
            response = QMessageBox.warning(
                    self, "Processing underway",
                    "We're still processing a video. "
                    "Are you sure you want to quit?",
                    QMessageBox.Cancel | QMessageBox.Yes, QMessageBox.Yes)
            wants_to_quit = (response == QMessageBox.Yes)

        if close_event is not None:
            if wants_to_quit:
                close_event.accept()
            else:
                close_event.ignore()        # prevents window from closing

        if wants_to_quit:
            self._thread.requestInterruption()  # see HEWorker.abort()
            self._thread.quit()             # stop worker thread's event loop
            self._thread.wait(5000)         # wait up to five seconds to stop
            self.app.quit()

    # -------------------------------------------------------------------------
    #  Media player
    # -------------------------------------------------------------------------

    def framenum(self):
        """
        Returns the frame number (integer) currently visible in the player
        """
        if self.current_video_item is None:
            return 0
        vc = self.current_video_item.vc
        return framenum_for_position(vc, vc.player.position())

    def set_framenum(self, framenum):
        """
        Sets the frame number in the currently-visible player.
        """
        # print('got to set_framenum, value = {}'.format(framenum))
        vc = self.current_video_item.vc
        if vc.notes is None:
            return
        framenum = min(max(0, framenum), vc.frames - 1)
        position = position_for_framenum(vc, framenum)
        vc.player.setPosition(position)

        # Update slider position. The media player doesn't always signal
        # positionChanged() while playback is paused.
        prev = self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(prev)

    def pause_movie(self):
        """
        Utility function to pause the movie if it's playing. It is careful to
        stop on a clean frame boundary, otherwise HEVideoView.paintEvent() can
        get off-by-one errors in the frame # when drawing overlays on top of
        the paused video.
        """
        if self.current_video_item is None:
            return
        vc = self.current_video_item.vc
        if vc.player.state() == QMediaPlayer.PlayingState:
            vc.player.pause()
            if vc.notes is not None:
                framenum = framenum_for_position(vc, vc.player.position())
                self.set_framenum(framenum)
                self.back_button.setEnabled(vc.has_played)
                self.forward_button.setEnabled(vc.has_played)

    def play_movie(self):
        """
        Utility function to un-pause the video.
        """
        if self.current_video_item is None:
            return
        vc = self.current_video_item.vc
        if vc.player.state() != QMediaPlayer.PlayingState:
            vc.player.play()
            self.back_button.setEnabled(False)
            self.forward_button.setEnabled(False)

            # Mark the video as having been played. This addresses an issue
            # where the step forward/backward actions don't work correctly
            # until the video has entered a playing state. (This issue occurs
            # on both Mac and Windows.)
            vc.has_played = True

    def repaint_player(self):
        """
        If the player is paused then force a repaint of the player. This is
        only necessary on Windows; the Mac OS player continuously repaints even
        when paused.
        """
        if platform.system() == 'Windows':
            if self.current_video_item is None:
                return
            if (self.current_video_item.vc.player.state()
                    != QMediaPlayer.PlayingState):
                # this is a hack but I haven't found any other method that
                # works, including self.view.repaint()
                self.play_movie()
                self.pause_movie()

    # -------------------------------------------------------------------------
    #  QMainWindow overrides
    # -------------------------------------------------------------------------

    def keyPressEvent(self, e):
        """
        We have to work around the fact that neither Qt nor Python can give
        us the instantaneous state of the keyboard, so we use key repeats to
        give us the desired function of the arrow keys which is to continue
        cueing frame-by-frame while a key is held. Do all of the repeated
        frame advancing in HEVideoView.paintEvent() so we don't get frame
        skips.

        keycodes at http://doc.qt.io/qt-5/qt.html#Key-enum
        """
        if (self.views_stacked_widget.currentIndex() != 0
                or self.current_video_item is None):
            super().keyPressEvent(e)
            return

        key = e.key()
        framenum = self.framenum()
        vc = self.current_video_item.vc
        notes = vc.notes
        viewitem = self.current_view_item

        if key == Qt.Key_Space and self.play_button.isEnabled():
            self.toggle_play()
        elif key == Qt.Key_Right and notes is not None and vc.has_played:
            # advance movie by one frame
            self.step_backward_until = None
            if self.step_forward_until is None:
                # not already playing forward
                self.step_forward_until = framenum + 1

                # this next line is necessary on Windows but not Mac OS, due to
                # differences in the way the video players work (the Mac player
                # calls paintEvent() continuously even when the player is
                # paused, while the Windows player does not). Keep it in for
                # cross-platform compatibility:
                self.step_forward()
            else:
                """
                Set target two frames ahead so that HEVideoView.paintEvent()
                won't stop the forward playing on the next time it executes.
                This gives smooth advancement whether the key repeat rate is
                faster or slower than the frame refresh rate, at the cost of
                overshooting by one frame.
                """
                self.step_forward_until = framenum + 2
        elif key == Qt.Key_Left and notes is not None and vc.has_played:
            # step back one frame
            self.step_forward_until = None
            if self.step_backward_until is None:
                self.step_backward_until = framenum - 1
                self.step_backward()     # see note above on step forward
            else:
                self.step_backward_until = framenum - 2
        elif key == Qt.Key_Down:
            # toggle overlays
            if notes is not None and notes['step'] >= 5:
                vc.overlays = not vc.overlays
                self.repaint_player()
        elif key == Qt.Key_Up:
            # toggle accuracy-related overlays
            if notes is not None and notes['step'] >= 5 and vc.overlays:
                active = not self.prefs['accuracy_overlay']
                self.prefs['ideal_points'] = active
                self.prefs['accuracy_overlay'] = active
                self.repaint_player()
        elif key == Qt.Key_K and notes is not None:
            # save a video clip of the currently-selected run
            if 'run' not in notes:
                return
            for i in range(self.view_list.count()):
                viewitem = self.view_list.item(i)
                if viewitem._type == 'run' and viewitem.isSelected():
                    self.sig_extract_clip.emit(vc.filepath, notes,
                                               viewitem._runindex)
                    self.worker_clipping_queue_length += 1
                    self.set_worker_busy_icon()
                    break
        elif key == Qt.Key_P and notes is not None:
            # process current video to analyze juggling content
            if 'step' not in notes or notes['step'] < 6:
                vc.doneprocessing = False
                self.sig_analyze_juggling.emit(vc.filepath, notes)
                self.worker_processing_queue_length += 1
                self.set_worker_busy_icon()
                self.current_video_item.setForeground(Qt.gray)
                self.build_view_list(self.current_video_item)
                # switch to 'scanner output' view
                for i in range(self.view_list.count()):
                    viewitem = self.view_list.item(i)
                    if viewitem._type == 'output':
                        viewitem.setSelected(True)
                        break
        elif key == Qt.Key_X and notes is not None:
            # play forward until next throw in run
            if 'arcs' not in notes or 'run' not in notes:
                return
            if viewitem is not None and viewitem._type == 'run':
                throwarcs = notes['run'][viewitem._runindex]['throw']
            else:
                throwarcs = notes['arcs']

            nextthrow = min((arc.f_throw for arc in throwarcs
                             if arc.f_throw > framenum + 1), default=None)
            if nextthrow is None:
                return
            self.step_forward_until = max(round(nextthrow), framenum + 1)
            self.step_backward_until = None
            self.step_forward()          # see note above on step forward
        elif key == Qt.Key_Z and notes is not None:
            # play backward until previous throw in run
            if 'arcs' not in notes or 'run' not in notes:
                return
            if viewitem is not None and viewitem._type == 'run':
                throwarcs = notes['run'][viewitem._runindex]['throw']
            else:
                throwarcs = notes['arcs']

            prevthrow = max((arc.f_throw for arc in throwarcs
                             if arc.f_throw < framenum - 1), default=None)
            if prevthrow is None:
                return
            self.step_forward_until = None
            self.step_backward_until = min(round(prevthrow), framenum - 1)
            self.step_backward()         # see note above on step forward
        elif key == Qt.Key_S and notes is not None:
            # go to next run
            if viewitem is not None and viewitem._type == 'run':
                row = self.view_list.row(viewitem)
                nextitem = self.view_list.item(row + 1)
                if nextitem is not None and nextitem._type == 'run':
                    prev = self.view_list.blockSignals(True)
                    self.view_list.setCurrentRow(row + 1)
                    self.view_list.blockSignals(prev)
                    self.current_view_item = nextitem
                    self.set_framenum(nextitem._startframe)
                    self.step_forward_until = None
                    self.step_backward_until = None
        elif key == Qt.Key_A and notes is not None:
            # go to previous run, or start of current run if we're more than
            # one second into playback
            if viewitem is not None and viewitem._type == 'run':
                if framenum > (viewitem._startframe + notes['fps']):
                    self.set_framenum(viewitem._startframe)
                    self.step_forward_until = None
                    self.step_backward_until = None
                else:
                    row = self.view_list.row(viewitem)
                    nextitem = self.view_list.item(row - 1)
                    if nextitem is not None and nextitem._type == 'run':
                        prev = self.view_list.blockSignals(True)
                        self.view_list.setCurrentRow(row - 1)
                        self.view_list.blockSignals(prev)
                        self.current_view_item = nextitem
                        self.set_framenum(nextitem._startframe)
                        self.step_forward_until = None
                        self.step_backward_until = None
        else:
            super().keyPressEvent(e)

    def sizeHint(self):
        return QSize(850, 600)

    def closeEvent(self, e):
        """
        Called when the user clicks the close icon in the window corner.
        """
        self.exit_call(e)

# -----------------------------------------------------------------------------


def framenum_for_position(videocontext, position):
    """
    Converts from position (milliseconds) to frame number.
    """
    if videocontext.notes is None:
        return None
    return floor(position * videocontext.notes['fps'] / 1000)


def position_for_framenum(videocontext, framenum):
    """
    Converts from frame number to position (milliseconds).
    """
    if videocontext.notes is None:
        return None
    fps = videocontext.notes['fps']
    return ceil(framenum * 1000 / fps) + floor(0.5 * 1000 / fps)
