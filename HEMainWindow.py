# HEMainWindow.py
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
                               QTableWidget, QTableWidgetItem, QMessageBox)
from PySide2.QtMultimedia import QMediaContent, QMediaPlayer

from HEWorker import HEWorker
from HEWidgets import (HEVideoView, HEVideoList, HEViewList,
                       HETableViewDelegate)


class HEMainWindow(QMainWindow):
    """
    The main application window.
    """
    CURRENT_APP_VERSION = "1.0"

    # signal that informs worker of new display preferences
    sig_new_prefs = Signal(dict)

    # signal that informs worker of a new video to process
    sig_process_video = Signal(str)

    # signal that informs worker to extract a clip from a video
    sig_extract_clip = Signal(str, dict, int)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle('Hawkeye')

        # currently-selected video and view in UI
        self.currentVideoItem = None
        self.currentViewItem = None

        # fast-stepping playback mode; see HEVideoView.do_playback_control()
        self.stepForwardUntil = None
        self.stepBackwardUntil = None

        self.prefs = {
            'markers': False,
            'throw_labels': True,
            'parabolas': True,
            'carries': True,
            'ideal_points': True,
            'accuracy_overlay': True,
            'torso': False,
            'resolution': 'Actual size'
        }

        # Media player on Mac OS bogs down on videos at 1080p resolution,
        # so downsample to 720 as a default.
        if platform.system() == 'Darwin':
            self.prefs['resolution'] = '720'

        self.makeUI()
        self.grabKeyboard()
        self.startWorker()

    # -------------------------------------------------------------------------
    #  User interface creation
    # -------------------------------------------------------------------------

    def makeUI(self):
        """
        Build all the UI elements in the application window.
        """
        self.makeMenus()
        lists_widget = self.makeListsWidget()
        player_widget = self.makePlayerWidget()
        prefs_widget = self.makePrefsWidget()
        output_widget = self.makeScannerOutputWidget()
        data_widget = self.makeDataWidget()
        about_widget = self.makeAboutWidget()

        # assemble window contents
        self.player_stackedWidget = QStackedWidget()
        self.player_stackedWidget.addWidget(player_widget)
        self.player_stackedWidget.addWidget(prefs_widget)
        self.player_stackedWidget.setCurrentIndex(0)

        self.views_stackedWidget = QStackedWidget()
        self.views_stackedWidget.addWidget(self.player_stackedWidget)
        self.views_stackedWidget.addWidget(output_widget)
        self.views_stackedWidget.addWidget(data_widget)
        self.views_stackedWidget.addWidget(about_widget)
        self.views_stackedWidget.setCurrentIndex(3)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(lists_widget)
        splitter.addWidget(self.views_stackedWidget)
        splitter.setCollapsible(1, False)

        self.setCentralWidget(splitter)

    def makeMenus(self):
        """
        Make the menu bar for the application.
        """
        openAction = QAction('&Open', self)
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open movie')
        openAction.triggered.connect(self.openFile)
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(openAction)
        fileMenu.addAction(exitAction)

    def makeListsWidget(self):
        """
        Widget for the left side of the UI, containing the Video and View
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
        self.busyIcon = QLabel()
        self.busyIcon.setSizePolicy(QSizePolicy.Preferred,
                                    QSizePolicy.Preferred)
        self.busyMovie = QMovie(os.path.join(base_dir,
                                             'resources/busy_icon.gif'))
        self.busyIcon.setMovie(self.busyMovie)
        video_label_layout.addWidget(self.busyIcon)
        self.busyIcon.hide()
        lists_layout.addLayout(video_label_layout)
        self.videoList = HEVideoList(self)
        self.videoList.itemSelectionChanged.connect(self.on_video_selected)
        self.videoList.setSizePolicy(QSizePolicy.Preferred,
                                     QSizePolicy.Expanding)
        lists_layout.addWidget(self.videoList)
        label = QLabel('View:')
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lists_layout.addWidget(label)
        self.viewList = HEViewList()
        self.viewList.itemSelectionChanged.connect(self.on_view_selected)
        self.viewList.setSizePolicy(QSizePolicy.Preferred,
                                    QSizePolicy.Expanding)
        lists_layout.addWidget(self.viewList)
        lists_widget = QWidget()
        lists_widget.setLayout(lists_layout)
        return lists_widget

    def makePlayerWidget(self):
        """
        The player widget is the area on the right, with the video player and
        control bar below it.
        """
        self.view = HEVideoView(self)
        self.view.setDragMode(QGraphicsView.NoDrag)

        self.zoomInButton = QPushButton('& + ', self)
        self.zoomInButton.setEnabled(False)
        self.zoomInButton.clicked.connect(self.zoomIn)

        self.zoomOutButton = QPushButton('& - ', self)
        # self.zoomOutButton = QPushButton('&\u2014', self)
        self.zoomOutButton.setEnabled(False)
        self.zoomOutButton.clicked.connect(self.zoomOut)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.togglePlay)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        # self.positionSlider.sliderMoved.connect(self.setPosition)
        self.positionSlider.valueChanged.connect(self.setPosition)

        rates = ['0.1', '0.25', '0.5', '1.0', '1.5', '2.0']
        self.playbackRate = QComboBox()
        for index, rate in enumerate(rates):
            self.playbackRate.insertItem(index, rate)
        self.playbackRate.setEditable(False)
        self.playbackRate.currentIndexChanged.connect(self.on_rate_change)
        self.playbackRate.setCurrentIndex(rates.index('0.5'))

        self.backButton = QPushButton()
        self.backButton.setEnabled(False)
        self.backButton.setIcon(self.style().standardIcon(
                QStyle.SP_MediaSkipBackward))
        self.backButton.clicked.connect(self.stepBackward)

        self.forwardButton = QPushButton()
        self.forwardButton.setEnabled(False)
        self.forwardButton.setIcon(self.style().standardIcon(
                QStyle.SP_MediaSkipForward))
        self.forwardButton.clicked.connect(self.stepForward)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.zoomInButton)
        controls_layout.addWidget(self.zoomOutButton)
        controls_layout.addWidget(self.playButton)
        controls_layout.addWidget(self.positionSlider)
        controls_layout.addWidget(self.playbackRate)
        controls_layout.addWidget(self.backButton)
        controls_layout.addWidget(self.forwardButton)

        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            base_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            base_dir = os.path.dirname(os.path.realpath(__file__))

        errorbar_layout = QHBoxLayout()
        errorbar_layout.setContentsMargins(0, 0, 7, 0)
        self.playerErrorLabel = QLabel()
        self.playerErrorLabel.setSizePolicy(QSizePolicy.Preferred,
                                            QSizePolicy.Maximum)
        aboutButton = QToolButton()
        aboutButton.setIcon(QIcon(
                os.path.join(base_dir, 'resources/about_icon.png')))
        aboutButton.setIconSize(QSize(20, 20))
        aboutButton.setStyleSheet('border: none;')
        aboutButton.clicked.connect(self.showAbout)
        prefsButton = QToolButton()
        prefsButton.setIcon(QIcon(
                os.path.join(base_dir, 'resources/preferences_icon.png')))
        prefsButton.setIconSize(QSize(20, 20))
        prefsButton.setStyleSheet('border: none;')
        prefsButton.clicked.connect(self.showPrefs)
        errorbar_layout.addWidget(self.playerErrorLabel)
        errorbar_layout.addWidget(aboutButton)
        errorbar_layout.addWidget(prefsButton)

        player_layout = QVBoxLayout()
        # layout_right_player.setContentsMargins(0, 10, 0, 0)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        player_layout.addWidget(self.view)
        player_layout.addLayout(controls_layout)
        player_layout.addLayout(errorbar_layout)
        player_widget = QWidget()
        player_widget.setLayout(player_layout)
        return player_widget

    def makePrefsWidget(self):
        """
        The preferences panel that is shown when the settings icon is clicked.
        """
        self.prefs_markers = QCheckBox('Ball detections')
        self.prefs_throwlabels = QCheckBox('Throw number labels')
        self.prefs_parabolas = QCheckBox('Ball arcs')
        self.prefs_carries = QCheckBox('Hand carry distances')
        self.prefs_ideal_points = QCheckBox(
                    'Ideal throw and catch points')
        self.prefs_accuracy_overlay = QCheckBox(
                    'Throwing accuracy overlay')
        self.prefs_torso = QCheckBox('Torso position')

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

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(10, 0, 30, 0)
        controls_layout.setAlignment(Qt.AlignVCenter)
        controls_layout.addWidget(QLabel('Video display options:'))
        controls_layout.addWidget(self.prefs_markers)
        controls_layout.addWidget(self.prefs_throwlabels)
        controls_layout.addWidget(self.prefs_parabolas)
        controls_layout.addWidget(self.prefs_carries)
        controls_layout.addWidget(self.prefs_ideal_points)
        controls_layout.addWidget(self.prefs_accuracy_overlay)
        controls_layout.addWidget(self.prefs_torso)
        controls_layout.addLayout(resolution_layout)
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Expanding)

        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.cancelPrefs)
        acceptButton = QPushButton('Accept')
        acceptButton.clicked.connect(self.setPrefs)
        buttons_layout.addWidget(cancelButton)
        buttons_layout.addWidget(acceptButton)
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

    def makeScannerOutputWidget(self):
        """
        A big text box and progress bar for showing output during the
        scanning process.
        """
        self.outputWidget = QPlainTextEdit()
        self.outputWidget.setReadOnly(True)
        self.outputWidget.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = self.outputWidget.font()
        font.setFamily('Courier')
        font.setStyleHint(QFont.Monospace)
        self.outputWidget.setFont(font)
        self.outputWidget.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Expanding)
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 0)

        output_layout = QVBoxLayout()
        label = QLabel('Scanner output:')
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        output_layout.addWidget(label)
        output_layout.addWidget(self.outputWidget)
        output_layout.addWidget(self.progressBar)
        output_widget = QWidget()
        output_widget.setLayout(output_layout)
        return output_widget

    def makeDataWidget(self):
        """
        A table showing relevant data for each throw detected in the video.
        This data can be exported to a CSV file.
        """
        data_layout = QVBoxLayout()
        self.dataWidget = QTableWidget()
        self.dataWidget.verticalHeader().setVisible(False)
        self.dataWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dataWidget.setSizePolicy(QSizePolicy.Expanding,
                                      QSizePolicy.Expanding)
        self.dataWidget.setShowGrid(False)
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignRight)
        exportButton = QPushButton('Export as CSV')
        exportButton.clicked.connect(self.exportData)
        button_layout.addWidget(exportButton)

        data_layout.addWidget(self.dataWidget)
        data_layout.addLayout(button_layout)
        data_widget = QWidget()
        data_widget.setLayout(data_layout)
        return data_widget

    def makeAboutWidget(self):
        """
        The info panel that is shown on startup, or when the user clicks the
        '?' icon in the lower right.
        """
        msg = ('<p><h2>Hawkeye Juggling Video Analyzer</h2></p>'
               '<p>Version ' + HEMainWindow.CURRENT_APP_VERSION + '</p>'
               '<p>Copyright &copy; 2019 Jack Boyce (jboyce@gmail.com)</p>'
               '<p>&nbsp;</p>'
               '<p>This software processes video files and tracks '
               'objects moving through the air in parabolic trajectories. '
               'Currently it detects balls only. For best results use video '
               'with a stationary camera, four or more objects, a high frame '
               'rate (60+ fps), minimum motion blur, and good brightness '
               'separation between the balls and background.</p>'
               '<p>Drop a video file onto the <b>Videos:</b> box to begin!</p>'
               '<p>Useful keyboard shortcuts when viewing video:</p>'
               '<ul>'
               '<li>space: toggle play/pause</li>'
               '<li>left/right: step backward/forward by one frame '
               '(hold to continue cueing)</li>'
               '<li>z, x: step backward/forward by one throw</li>'
               '<li>a, s: step backward/forward by one run</li>'
               '<li>down: toggle overlays</li>'
               '<li>up: toggle accuracy-related overlays</li>'
               '<li>k: save video clip of current run</li>'
               '</ul>'
               '<p>&nbsp;</p>'
               '<p><small>All portions of this software written by Jack Boyce '
               'are provided under the MIT License. Other software '
               'distributed as part of Hawkeye is done so under the terms of '
               'their respective non-commercial licenses: '
               'OpenCV version 3.4.4 (3-clause BSD License), '
               'YOLO-tiny version 3 (MIT License), '
               'Qt version 5.6.2 (LGPL v2.1), '
               'PySide2 version 5.6.0 (LGPL v2.1), '
               'FFmpeg version 3.4.2 (LGPL v2.1).</small></p>'
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
        okButton = QPushButton('OK')
        okButton.clicked.connect(self.cancelAbout)
        button_layout.addWidget(okButton)
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
    #  Worker thread
    # -------------------------------------------------------------------------

    def startWorker(self):
        """
        Starts up worker object on a separate thread, to handle the lengthy
        processing operations without blocking the UI.

        For data consistency all communication with the worker is done via Qt
        signals and slots.
        """
        self._worker = worker = HEWorker(self.app)
        self._thread = thread = QThread()
        worker.moveToThread(thread)

        # signals from UI thread to worker
        self.sig_new_prefs.connect(worker.on_new_prefs)
        self.sig_process_video.connect(worker.on_process_video)
        self.sig_extract_clip.connect(worker.on_extract_clip)

        # signals from worker back to UI thread
        worker.sig_progress.connect(self.on_worker_step)
        worker.sig_output.connect(self.on_worker_output)
        worker.sig_error.connect(self.on_worker_error)
        worker.sig_video_done.connect(self.on_worker_displayvid_done)
        worker.sig_notes_done.connect(self.on_worker_notes_done)
        worker.sig_clipping_done.connect(self.on_worker_clipping_done)

        # have worker thread clean up worker object on close; otherwise we
        # can get a "QObject::~QObject: Timers cannot be stopped from another
        # thread" console error
        thread.finished.connect(worker.deleteLater)

        thread.start(QThread.LowPriority)   # start thread's event loop

        self.sig_new_prefs.emit(self.prefs)
        self.worker_processing_queue_length = 0
        self.worker_clipping_queue_length = 0
        self.setWorkerBusyIcon()

    @Slot(str, int, int)
    def on_worker_step(self, file_id: str, step: int, stepstotal: int):
        """
        Signaled by worker when there is a progress update on processing
        a video.
        """
        for i in range(0, self.videoList.count()):
            item = self.videoList.item(i)
            if item is None:
                continue
            if item.vc.filepath == file_id:
                if item.vc.doneprocessing is False:
                    item.vc.processing_step = step
                    item.vc.processing_steps_total = stepstotal
                    if item is self.currentVideoItem:
                        self.progressBar.setValue(step)
                        self.progressBar.setMaximum(stepstotal)
                break

    @Slot(str, str)
    def on_worker_output(self, file_id: str, output: str):
        """
        Signaled by worker when there is text output from processing.
        """
        for i in range(0, self.videoList.count()):
            item = self.videoList.item(i)
            if item is None:
                continue
            if item.vc.filepath == file_id:
                # continue to take output even if processing is done
                item.vc.output += output
                if item is self.currentVideoItem:
                    self.outputWidget.moveCursor(QTextCursor.End)
                    self.outputWidget.insertPlainText(output)
                    self.outputWidget.moveCursor(QTextCursor.End)
                break

    @Slot(str, str)
    def on_worker_error(self, file_id: str, errormsg: str):
        """
        Signaled by worker when there is an error processing `file_id`.

        When file_id is an empty string it signifies a general nonfatal error
        in the worker thread, which we show to the user.
        """
        if file_id != "":
            for i in range(0, self.videoList.count()):
                item = self.videoList.item(i)
                if item is None:
                    continue
                if item.vc.filepath == file_id:
                    if item.vc.doneprocessing is False:
                        # only do these steps on the first error for the item
                        self.worker_processing_queue_length -= 1
                        self.setWorkerBusyIcon()
                        item.vc.notes = None
                        item.vc.doneprocessing = True
                        item.setForeground(item.foreground)
                        if item is self.currentVideoItem:
                            self.buildViewList(item)
                            self.progressBar.hide()
                    break
        else:
            QMessageBox.warning(self, 'Oops, we encountered a problem.',
                                errormsg, QMessageBox.Ok, QMessageBox.Ok)

    @Slot(str, dict, int, dict)
    def on_worker_displayvid_done(self, file_id: str, fileinfo: dict,
                                  resolution: int, notes: dict):
        """
        Signaled by worker when the transcoded display version of the video is
        completed. This will always come before video notes completion below.
        """
        for i in range(self.videoList.count()):
            item = self.videoList.item(i)
            if item is None:
                continue
            if item.vc.filepath == file_id:
                if item.vc.doneprocessing is False:
                    # only do these steps once, and only if there was not
                    # previously an error
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
                        def mapToDisplayVideo(x, y):
                            return x, y
                    else:
                        orig_height = notes['frame_height']
                        orig_width = notes['frame_width']
                        display_height = item.vc.videoresolution
                        display_width = round(orig_width * display_height
                                              / orig_height)
                        if display_width % 2 == 1:
                            display_width += 1

                        def mapToDisplayVideo(x, y):
                            new_x = x * display_width / orig_width
                            new_y = y * display_height / orig_height
                            return new_x, new_y
                    item.vc.map = mapToDisplayVideo

                    # load the video into the player
                    item.vc.player.pause()
                    item.vc.player.setMedia(QMediaContent(
                            QUrl.fromLocalFile(item.vc.videopath)))
                    item.vc.player.setPosition(0)
                    item.vc.player.pause()

                    if item is self.currentVideoItem:
                        self.playButton.setEnabled(True)
                        # block signals so we don't trigger setPosition()
                        prev = self.positionSlider.blockSignals(True)
                        self.positionSlider.setRange(0, item.vc.duration)
                        self.positionSlider.setValue(0)
                        self.positionSlider.blockSignals(prev)
                        self.buildViewList(item)

                break

    @Slot(str, dict)
    def on_worker_notes_done(self, file_id: str, notes: dict):
        """
        Signaled by worker when notes for the video is completed. This will
        always come after the display video completion.
        """
        for i in range(self.videoList.count()):
            item = self.videoList.item(i)
            if item is None:
                continue
            if item.vc.filepath == file_id:
                if item.vc.doneprocessing is False:
                    # only do these steps once, and only if there was not
                    # previously an error
                    self.worker_processing_queue_length -= 1
                    self.setWorkerBusyIcon()
                    item.setForeground(item.foreground)

                    item.vc.doneprocessing = True
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
                        item.vc.position = positionForFramenum(item.vc,
                                                               startframe)
                        item.vc.player.setPosition(item.vc.position)
                        item.vc.player.pause()      # necessary on Mac?

                    if item is self.currentVideoItem:
                        self.progressBar.hide()
                        # block signals so we don't trigger setPosition()
                        prev = self.positionSlider.blockSignals(True)
                        self.positionSlider.setRange(0, item.vc.duration)
                        self.positionSlider.setValue(item.vc.position)
                        self.positionSlider.blockSignals(prev)
                        self.buildViewList(item)
                        # switch to movie view
                        self.views_stackedWidget.setCurrentIndex(0)

                break

    @Slot()
    def on_worker_clipping_done(self):
        self.worker_clipping_queue_length -= 1
        self.setWorkerBusyIcon()

    def isWorkerBusy(self):
        """
        Returns True when worker is busy, False when idle.
        """
        return (self.worker_processing_queue_length
                + self.worker_clipping_queue_length != 0)

    def setWorkerBusyIcon(self):
        """
        Called whenever the state of the worker changes. We use this hook to
        show/hide the "worker busy" icon in the UI.
        """
        if self.isWorkerBusy():
            self.busyIcon.show()
            self.busyMovie.start()
        else:
            self.busyIcon.hide()
            self.busyMovie.stop()

    # -------------------------------------------------------------------------
    #  UI interactions
    # -------------------------------------------------------------------------

    @Slot()
    def on_video_selected(self):
        """
        Called when a video is selected in the HEVideoList, either through
        user action or programmatically (the latter happens when a file is
        dropped on the HEVideoList).
        """
        items = self.videoList.selectedItems()
        if len(items) == 0:
            return
        item = items[0]
        if item is self.currentVideoItem:
            return

        # pause video player for current movie, if one is showing
        self.pauseMovie()

        self.currentVideoItem = item

        # set up video viewer
        # attach HEVideoView to this video's media player:
        self.view.setScene(item.vc.graphicsscene)
        item.vc.player.setPlaybackRate(float(
                    self.playbackRate.currentText()))
        self.stepForwardUntil = None
        self.stepBackwardUntil = None

        # set up UI elements
        self.playButton.setEnabled(item.vc.videopath is not None)
        self.backButton.setEnabled(item.vc.has_played)
        self.forwardButton.setEnabled(item.vc.has_played)
        self.playerErrorLabel.setText('')
        if item.vc.duration is not None:
            # block signals so we don't trigger setPosition()
            prev = self.positionSlider.blockSignals(True)
            self.positionSlider.setRange(0, item.vc.duration)
            self.positionSlider.setValue(item.vc.position)
            self.positionSlider.blockSignals(prev)

        self.buildViewList(item)

        if item.vc.doneprocessing and item.vc.videopath is not None:
            # switch to video view
            self.views_stackedWidget.setCurrentIndex(0)
        else:
            # switch to 'scanner output' view
            for i in range(self.viewList.count()):
                viewitem = self.viewList.item(i)
                if viewitem._type == 'output':
                    viewitem.setSelected(True)
                    break

    def buildViewList(self, videoitem):
        """
        Construct the set of views to make available for a given video. This
        populates the View list in the UI.
        """
        notes = videoitem.vc.notes
        self.viewList.clear()

        if (not videoitem.vc.doneprocessing
                and videoitem.vc.videopath is not None):
            # display video is ready but notes processing still underway
            headeritem = QListWidgetItem('')
            headeritem._type = 'video'
            headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
            header = QLabel('Video')
            self.viewList.addItem(headeritem)
            self.viewList.setItemWidget(headeritem, header)

        if videoitem.vc.doneprocessing and notes is not None:
            if notes['runs'] > 0:
                headeritem = QListWidgetItem('')
                headeritem._type = 'blank'
                headeritem.setFlags(headeritem.flags() & ~Qt.ItemIsSelectable)
                header = QLabel('Run #, Balls, Throws')
                self.viewList.addItem(headeritem)
                self.viewList.setItemWidget(headeritem, header)

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
                    runitem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
                    run = QLabel('{}, {}, {}'. format(i + 1,
                                                      run_dict['balls'],
                                                      run_dict['throws']))
                    self.viewList.addItem(runitem)
                    self.viewList.setItemWidget(runitem, run)
            else:
                headeritem = QListWidgetItem('')
                headeritem._type = 'blank'
                headeritem.setFlags(headeritem.flags() & ~Qt.ItemIsSelectable)
                header = QLabel('(no runs detected)')
                self.viewList.addItem(headeritem)
                self.viewList.setItemWidget(headeritem, header)

            headeritem = QListWidgetItem('')
            headeritem._type = 'blank'
            headeritem.setFlags(headeritem.flags() & ~Qt.ItemIsSelectable)
            self.viewList.addItem(headeritem)

            headeritem = QListWidgetItem('')
            headeritem._type = 'video'
            headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
            header = QLabel('Video')
            self.viewList.addItem(headeritem)
            self.viewList.setItemWidget(headeritem, header)

            headeritem = QListWidgetItem('')
            headeritem._type = 'data'
            headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
            header = QLabel('Run data')
            self.viewList.addItem(headeritem)
            self.viewList.setItemWidget(headeritem, header)

        if not (videoitem.vc.doneprocessing and len(videoitem.vc.output) == 0):
            headeritem = QListWidgetItem('')
            headeritem._type = 'output'
            headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
            header = QLabel('Scanner output')
            self.viewList.addItem(headeritem)
            self.viewList.setItemWidget(headeritem, header)

        if not videoitem.vc.doneprocessing:
            headeritem = QListWidgetItem('')
            headeritem._type = 'blank'
            headeritem.setFlags(headeritem.flags() & ~Qt.ItemIsSelectable)
            header = QLabel('(processing...)')
            header.setStyleSheet('QLabel { color : gray }')
            self.viewList.addItem(headeritem)
            self.viewList.setItemWidget(headeritem, header)

    @Slot()
    def on_view_selected(self):
        """
        Called when an item is selected in the View list, either through user
        action or programmatically.
        """
        items = self.viewList.selectedItems()
        if len(items) == 0:
            return
        item = items[0]
        self.currentViewItem = item

        if item._type == 'video':
            self.views_stackedWidget.setCurrentIndex(0)
        elif item._type == 'output':
            self.pauseMovie()
            self.outputWidget.setPlainText(self.currentVideoItem.vc.output)
            if self.currentVideoItem.vc.doneprocessing:
                self.progressBar.hide()
            else:
                # print("calling ensureCursorVisible()")
                self.outputWidget.moveCursor(QTextCursor.End)
                self.outputWidget.ensureCursorVisible()
                # vsb = self.outputWidget.verticalScrollBar()
                # if vsb is not None:
                #    vsb.setValue(vsb.maximum())
                self.progressBar.setValue(
                        self.currentVideoItem.vc.processing_step)
                self.progressBar.setMaximum(
                        self.currentVideoItem.vc.processing_steps_total)
                self.progressBar.show()
            self.views_stackedWidget.setCurrentIndex(1)
        elif item._type == 'data':
            notes = self.currentVideoItem.vc.notes
            if notes is not None:
                self.pauseMovie()
                self.fillDataView(notes)
                self.views_stackedWidget.setCurrentIndex(2)
        elif item._type == 'run':
            self.views_stackedWidget.setCurrentIndex(0)
            self.playMovie()
            self.setFramenum(item._startframe)
        else:
            # shouldn't ever get here
            pass

    def fillDataView(self, notes):
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

        self.dataWidget.setRowCount(len(notes['arcs']))
        self.dataWidget.setColumnCount(len(headers))
        for col, name in enumerate(headers):
            item = QTableWidgetItem(name)
            item.setTextAlignment(alignments[col] | Qt.AlignVCenter)
            self.dataWidget.setHorizontalHeaderItem(col, item)

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
                    torso_x = (x + 0.5 * w) * notes['cm_per_pixel']
                    torso_x_str = f'{torso_x:.1f}'
                    throw_x -= torso_x
                    catch_x -= torso_x
                except KeyError:
                    torso_x_str = '-'
                prev_id = '-' if arc.prev is None else str(arc.prev.throw_id)
                next_id = '-' if arc.next is None else str(arc.next.throw_id)

                self.dataWidget.setItem(row, 0, QTableWidgetItem(
                        str(arc.run_id)))
                self.dataWidget.setItem(row, 1, QTableWidgetItem(
                        str(arc.throw_id)))
                self.dataWidget.setItem(row, 2, QTableWidgetItem(hands))
                self.dataWidget.setItem(row, 3, QTableWidgetItem(
                        f'{throw_x:.1f}'))
                self.dataWidget.setItem(row, 4, QTableWidgetItem(
                        f'{catch_x:.1f}'))
                self.dataWidget.setItem(row, 5, QTableWidgetItem(
                        f'{angle:.2f}'))
                self.dataWidget.setItem(row, 6, QTableWidgetItem(
                        f'{height:.1f}'))
                self.dataWidget.setItem(row, 7, QTableWidgetItem(
                        f'{throw_t:.3f}'))
                self.dataWidget.setItem(row, 8, QTableWidgetItem(
                        f'{catch_t:.3f}'))
                self.dataWidget.setItem(row, 9, QTableWidgetItem(torso_x_str))
                self.dataWidget.setItem(row, 10, QTableWidgetItem(prev_id))
                self.dataWidget.setItem(row, 11, QTableWidgetItem(next_id))
                for col, align in enumerate(alignments):
                    item = self.dataWidget.item(row, col)
                    item.setTextAlignment(align | Qt.AlignVCenter)
                    if throw_hand == 'R':
                        item.setBackground(right_throws_color)
                row += 1
            if run_id < notes['runs']:
                boundary_ix.append(row)

        # item delegate draws lines between runs:
        self.dataWidget.setItemDelegate(HETableViewDelegate(self.dataWidget,
                                                            boundary_ix))
        self.dataWidget.resizeColumnsToContents()
        self.dataWidget.resizeRowsToContents()

    @Slot()
    def exportData(self):
        """
        Called when the user clicks the 'Export as CSV' button in data view.
        """
        data = []
        line = []
        for col in range(self.dataWidget.columnCount()):
            line.append(str(self.dataWidget.horizontalHeaderItem(col).text()))
        data.append(', '.join(line))
        for row in range(self.dataWidget.rowCount()):
            line = []
            for col in range(self.dataWidget.columnCount()):
                line.append(str(self.dataWidget.item(row, col).text()))
            data.append(', '.join(line))
        output = '\n'.join(data)

        csvfile_path = self.currentVideoItem.vc.csvpath
        filename = QFileDialog.getSaveFileName(self, 'Save CSV File',
                                               csvfile_path)
        if filename != '':
            filepath = os.path.abspath(filename[0])
            with open(filepath, 'w') as f:
                f.write(output)

    @Slot()
    def showPrefs(self):
        """
        Called when the user clicks the preferences icon in the UI.
        """
        self.pauseMovie()
        self.prefs_markers.setChecked(self.prefs['markers'])
        self.prefs_throwlabels.setChecked(self.prefs['throw_labels'])
        self.prefs_parabolas.setChecked(self.prefs['parabolas'])
        self.prefs_carries.setChecked(self.prefs['carries'])
        self.prefs_ideal_points.setChecked(self.prefs['ideal_points'])
        self.prefs_accuracy_overlay.setChecked(self.prefs['accuracy_overlay'])
        self.prefs_torso.setChecked(self.prefs['torso'])
        self.prefs_resolution.setCurrentText(self.prefs['resolution'])
        self.player_stackedWidget.setCurrentIndex(1)

    @Slot()
    def setPrefs(self):
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
        self.prefs['torso'] = self.prefs_torso.isChecked()
        self.prefs['resolution'] = self.prefs_resolution.currentText()
        self.sig_new_prefs.emit(self.prefs)
        self.player_stackedWidget.setCurrentIndex(0)

        # If display resolution changed, reprocess all videos through worker
        # to create the needed video files. Do the current video first.
        if self.prefs['resolution'] != old_resolution:
            if self.currentVideoItem is not None:
                self.currentVideoItem.vc.doneprocessing = False
                self.sig_process_video.emit(self.currentVideoItem.vc.filepath)
                self.worker_processing_queue_length += 1

                # auto-select 'scanner output' view
                for i in range(self.viewList.count()):
                    viewitem = self.viewList.item(i)
                    if viewitem._type == 'output':
                        viewitem.setSelected(True)
                        break

            for i in range(self.videoList.count()):
                item = self.videoList.item(i)
                if item is not None and item is not self.currentVideoItem:
                    item.vc.doneprocessing = False
                    self.sig_process_video.emit(item.vc.filepath)
                    self.worker_processing_queue_length += 1

            self.setWorkerBusyIcon()

    @Slot()
    def cancelPrefs(self):
        """
        Called when the user clicks "Cancel" in the preferences panel.
        """
        self.player_stackedWidget.setCurrentIndex(0)

    @Slot()
    def showAbout(self):
        """
        Called when the user clicks the about icon in the UI.
        """
        self.pauseMovie()
        self.views_stackedWidget.setCurrentIndex(3)

    @Slot()
    def cancelAbout(self):
        """
        Called when the user clicks 'OK' in the about panel.
        """
        self.views_stackedWidget.setCurrentIndex(0)

    @Slot()
    def zoomIn(self):
        """
        Called when the user clicks the zoom in button in the UI.
        """
        self.view.scale(1.2, 1.2)
        self.zoomOutButton.setEnabled(True)
        self.view.videosnappedtoframe = False
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

    @Slot()
    def zoomOut(self):
        """
        Called when the user clicks the zoom out button in the UI.
        """
        if not self.view.videosnappedtoframe:
            self.view.scale(1/1.2, 1/1.2)

            # see if the entire frame is now visible; if so then snap to frame
            portRect = self.view.viewport().rect()
            scenePolygon = self.view.mapToScene(portRect)
            graphicsvideoitem = self.currentVideoItem.vc.graphicsvideoitem
            itemPolygon = graphicsvideoitem.mapFromScene(scenePolygon)
            itemRect = itemPolygon.boundingRect()
            if itemRect.contains(graphicsvideoitem.boundingRect()):
                self.view.fitInView(graphicsvideoitem.boundingRect(),
                                    Qt.KeepAspectRatio)
                self.zoomOutButton.setEnabled(False)
                self.view.videosnappedtoframe = True
                self.view.setDragMode(QGraphicsView.NoDrag)

    @Slot()
    def stepBackward(self):
        """
        Called when the user clicks the 'back one frame' button in the UI.
        """
        if self.currentVideoItem.vc.notes is None:
            return
        self.currentVideoItem.vc.player.pause()
        self.setFramenum(self.framenum() - 1)

    @Slot()
    def stepForward(self):
        """
        Called when the user clicks the 'forward one frame' button in the UI.
        """
        if self.currentVideoItem.vc.notes is None:
            return
        self.currentVideoItem.vc.player.pause()
        self.setFramenum(self.framenum() + 1)

    @Slot(int)
    def on_rate_change(self, index: int):
        """
        Called when the user selects a playback rate on the dropdown menu.
        """
        if self.currentVideoItem is None:
            return
        self.currentVideoItem.vc.player.setPlaybackRate(float(
                    self.playbackRate.currentText()))

    @Slot()
    def openFile(self):
        """
        Called when the user chooses 'Open Video' from the File menu.
        """
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Video',
                                                  QDir.homePath())
        if filename != '':
            filepath = os.path.abspath(filename)
            lastitem = self.videoList.addVideo(filepath)
            if lastitem is not None:
                lastitem.setSelected(True)

    @Slot()
    def togglePlay(self):
        """
        Called when the user clicks the play/pause button in the UI.
        """
        self.stepBackwardUntil = self.stepForwardUntil = None

        if (self.currentVideoItem.vc.player.state() ==
                QMediaPlayer.PlayingState):
            self.pauseMovie()
        else:
            self.playMovie()

    @Slot(int)
    def setPosition(self, position):
        """
        Called when user interacts with the position slider.

        This is NOT called when the position slider changes as a result of
        playing the movie. See HEVideoView.do_playback_control().
        """
        vc = self.currentVideoItem.vc
        if vc.notes is not None:
            self.setFramenum(framenumForPosition(vc, position))
        else:
            vc.player.setPosition(position)

    @Slot()
    def exitCall(self, close_event=None):
        """
        Called when the user selects Quit in the menu, or clicks the close
        box in the window's corner. This does a clean exit of the application.
        """
        self.pauseMovie()
        wants_to_quit = True

        if self.isWorkerBusy():
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
        if self.currentVideoItem is None:
            return 0
        vc = self.currentVideoItem.vc
        return framenumForPosition(vc, vc.player.position())

    def setFramenum(self, framenum):
        """
        Sets the frame number in the currently-visible player.
        """
        # print('got to setFramenum, value = {}'.format(framenum))
        vc = self.currentVideoItem.vc
        if vc.notes is None:
            return
        framenum = min(max(0, framenum), vc.frames - 1)
        position = positionForFramenum(vc, framenum)
        vc.player.setPosition(position)

        # Update slider position. The media player doesn't always signal
        # positionChanged() while playback is paused.
        prev = self.positionSlider.blockSignals(True)
        self.positionSlider.setValue(position)
        self.positionSlider.blockSignals(prev)

    def pauseMovie(self):
        """
        Utility function to pause the movie if it's playing. It is careful to
        stop on a clean frame boundary, otherwise HEVideoView.paintEvent() can
        get off-by-one errors in the frame # when drawing overlays on top of
        the paused video.
        """
        if self.currentVideoItem is None:
            return
        vc = self.currentVideoItem.vc
        if vc.player.state() == QMediaPlayer.PlayingState:
            vc.player.pause()
            if vc.notes is not None:
                framenum = framenumForPosition(vc, vc.player.position())
                self.setFramenum(framenum)
                self.backButton.setEnabled(vc.has_played)
                self.forwardButton.setEnabled(vc.has_played)

    def playMovie(self):
        """
        Utility function to un-pause the video.
        """
        if self.currentVideoItem is None:
            return
        vc = self.currentVideoItem.vc
        if vc.player.state() != QMediaPlayer.PlayingState:
            vc.player.play()
            self.backButton.setEnabled(False)
            self.forwardButton.setEnabled(False)

            # Mark the video as having been played. This addresses an issue
            # where the step forward/backward actions don't work correctly
            # until the video has entered a playing state. (This issue occurs
            # on both Mac and Windows.)
            vc.has_played = True

    def repaintPlayer(self):
        """
        If the player is paused then force a repaint of the player. This is
        only necessary on Windows; the Mac OS player continuously repaints even
        when paused.
        """
        if platform.system() == 'Windows':
            if self.currentVideoItem is None:
                return
            if (self.currentVideoItem.vc.player.state()
                    != QMediaPlayer.PlayingState):
                # this is a hack but I haven't found any other method that
                # works, including self.view.repaint()
                self.playMovie()
                self.pauseMovie()

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
        if (self.views_stackedWidget.currentIndex() != 0
                or self.currentVideoItem is None):
            super().keyPressEvent(e)
            return

        key = e.key()
        framenum = self.framenum()
        vc = self.currentVideoItem.vc
        notes = vc.notes
        viewitem = self.currentViewItem

        if key == Qt.Key_Space and self.playButton.isEnabled():
            self.togglePlay()
        elif key == Qt.Key_Right and notes is not None and vc.has_played:
            # advance movie by one frame
            self.stepBackwardUntil = None
            if self.stepForwardUntil is None:
                # not already playing forward
                self.stepForwardUntil = framenum + 1

                # this next line is necessary on Windows but not Mac OS, due to
                # differences in the way the video players work (the Mac player
                # calls paintEvent() continuously even when the player is
                # paused, while the Windows player does not). Keep it in for
                # cross-platform compatibility:
                self.stepForward()
            else:
                """
                Set target two frames ahead so that HEVideoView.paintEvent()
                won't stop the forward playing on the next time it executes.
                This gives smooth advancement whether the key repeat rate is
                faster or slower than the frame refresh rate, at the cost of
                overshooting by one frame.
                """
                self.stepForwardUntil = framenum + 2
        elif key == Qt.Key_Left and notes is not None and vc.has_played:
            # step back one frame
            self.stepForwardUntil = None
            if self.stepBackwardUntil is None:
                self.stepBackwardUntil = framenum - 1
                self.stepBackward()     # see note above on step forward
            else:
                self.stepBackwardUntil = framenum - 2
        elif key == Qt.Key_Down:
            # toggle overlays
            if notes is not None and notes['step'] >= 5:
                vc.overlays = not vc.overlays
                self.repaintPlayer()
        elif key == Qt.Key_Up:
            # toggle accuracy-related overlays
            if notes is not None and notes['step'] >= 5:
                active = not self.prefs['accuracy_overlay']
                self.prefs['ideal_points'] = active
                self.prefs['accuracy_overlay'] = active
                self.repaintPlayer()
        elif key == Qt.Key_K and notes is not None:
            # save a video clip of the currently-selected run
            if 'run' not in notes:
                return
            for i in range(self.viewList.count()):
                viewitem = self.viewList.item(i)
                if viewitem._type == 'run' and viewitem.isSelected():
                    self.sig_extract_clip.emit(vc.filepath, notes,
                                               viewitem._runindex)
                    self.worker_clipping_queue_length += 1
                    self.setWorkerBusyIcon()
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
            self.stepForwardUntil = max(round(nextthrow), framenum + 1)
            self.stepBackwardUntil = None
            self.stepForward()          # see note above on step forward
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
            self.stepForwardUntil = None
            self.stepBackwardUntil = min(round(prevthrow), framenum - 1)
            self.stepBackward()         # see note above on step forward
        elif key == Qt.Key_S and notes is not None:
            # go to next run
            if viewitem is not None and viewitem._type == 'run':
                row = self.viewList.row(viewitem)
                nextitem = self.viewList.item(row + 1)
                if nextitem is not None and nextitem._type == 'run':
                    prev = self.viewList.blockSignals(True)
                    self.viewList.setCurrentRow(row + 1)
                    self.viewList.blockSignals(prev)
                    self.currentViewItem = nextitem
                    self.setFramenum(nextitem._startframe)
                    self.stepForwardUntil = None
                    self.stepBackwardUntil = None
        elif key == Qt.Key_A and notes is not None:
            # go to previous run, or start of current run if we're more than
            # one second into playback
            if viewitem is not None and viewitem._type == 'run':
                if framenum > (viewitem._startframe + notes['fps']):
                    self.setFramenum(viewitem._startframe)
                    self.stepForwardUntil = None
                    self.stepBackwardUntil = None
                else:
                    row = self.viewList.row(viewitem)
                    nextitem = self.viewList.item(row - 1)
                    if nextitem is not None and nextitem._type == 'run':
                        prev = self.viewList.blockSignals(True)
                        self.viewList.setCurrentRow(row - 1)
                        self.viewList.blockSignals(prev)
                        self.currentViewItem = nextitem
                        self.setFramenum(nextitem._startframe)
                        self.stepForwardUntil = None
                        self.stepBackwardUntil = None
        else:
            super().keyPressEvent(e)

    def sizeHint(self):
        return QSize(850, 600)

    def closeEvent(self, e):
        """
        Called when the user clicks the close icon in the window corner.
        """
        self.exitCall(e)

# -----------------------------------------------------------------------------


def framenumForPosition(videocontext, position):
    """
    Converts from position (milliseconds) to frame number.
    """
    if videocontext.notes is None:
        return None
    return floor(position * videocontext.notes['fps'] / 1000)


def positionForFramenum(videocontext, framenum):
    """
    Converts from frame number to position (milliseconds).
    """
    if videocontext.notes is None:
        return None
    fps = videocontext.notes['fps']
    return ceil(framenum * 1000 / fps) + floor(0.5 * 1000 / fps)
