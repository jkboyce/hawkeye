# Hawkeye.py
#
# Hawkeye application window and main entry point.
#
# Copyright 2018 Jack Boyce (jboyce@gmail.com)

import os
import sys
import time
import platform
from math import log10, ceil, floor, atan, pi

from PySide2.QtCore import (QDir, QSize, Qt, QUrl, QPoint, QPointF, QThread,
                            Signal, Slot)
from PySide2.QtGui import (QPainter, QFont, QTextCursor, QPainterPath, QIcon,
                           QPen, QColor)
from PySide2.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QSlider, QStyle,
                               QVBoxLayout, QWidget, QGraphicsView,
                               QGraphicsScene, QListWidget, QListWidgetItem,
                               QSplitter, QStackedWidget, QPlainTextEdit,
                               QProgressBar, QMainWindow, QAction, QComboBox,
                               QAbstractItemView, QToolButton, QCheckBox,
                               QTableWidget, QTableWidgetItem,
                               QStyledItemDelegate)
from PySide2.QtMultimedia import QMediaContent, QMediaPlayer
from PySide2.QtMultimediaWidgets import QGraphicsVideoItem

from HEWorker import HEWorker


class HEMainWindow(QMainWindow):
    """
    The main application window.
    """

    # signal that informs the worker of a new video to process
    sig_new_work = Signal(str)

    # signal that informs the worker of new display preferences
    sig_new_prefs = Signal(dict)

    # signal that tells the worker to initiate application quit
    sig_worker_quit = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Hawkeye')

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handlePlayerError)

        self.makeUI()

        self.prefs = {
            'markers': True,
            'torso': True,
            'parabolas': True,
            'throw_labels': True,
            'ideal_throws': False,
            'resolution': 'Actual size'
        }

        self.grabKeyboard()
        self.currentVideoItem = None
        self.currentViewItem = None
        self.playForwardUntil = None
        self.playBackwardUntil = None

        self.startWorker()

    def makeUI(self):
        """
        Build all the UI elements in the application window.
        """
        self.makeMenus()
        lists_widget = self.makeListsWidget()
        player_widget = self.makePlayerWidget()
        prefs_widget = self.makePrefsWidget()
        output_widget = self.makeScannerOutputWidget()
        stats_widget = self.makeStatsWidget()
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
        self.views_stackedWidget.addWidget(stats_widget)
        self.views_stackedWidget.addWidget(data_widget)
        self.views_stackedWidget.addWidget(about_widget)
        self.views_stackedWidget.setCurrentIndex(4)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(lists_widget)
        splitter.addWidget(self.views_stackedWidget)
        splitter.setCollapsible(1, False)

        self.setCentralWidget(splitter)

    def makeMenus(self):
        # menu bar
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
        lists_layout = QVBoxLayout()
        label = QLabel('Videos:')
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lists_layout.addWidget(label)
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
        self.item = QGraphicsVideoItem()
        self.scene = QGraphicsScene(self.view)
        self.scene.addItem(self.item)
        self.view.setScene(self.scene)
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

        # self.backButton = QPushButton('&<', self)
        self.backButton = QPushButton()
        self.backButton.setEnabled(False)
        self.backButton.setIcon(self.style().standardIcon(
                QStyle.SP_MediaSkipBackward))
        self.backButton.clicked.connect(self.stepBack)

        # self.forwardButton = QPushButton('&>', self)
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

        errorbar_layout = QHBoxLayout()
        errorbar_layout.setContentsMargins(0, 0, 7, 0)
        self.playerErrorLabel = QLabel()
        # self.playerErrorLabel.setText('This is where error output goes.')
        self.playerErrorLabel.setSizePolicy(QSizePolicy.Preferred,
                                            QSizePolicy.Maximum)
        aboutButton = QToolButton()
        aboutButton.setIcon(QIcon('about_icon.png'))
        aboutButton.setIconSize(QSize(20, 20))
        aboutButton.setStyleSheet('border: none;')
        aboutButton.clicked.connect(self.showAbout)
        prefsButton = QToolButton()
        prefsButton.setIcon(QIcon('preferences_icon.png'))
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
        self.prefs_torso = QCheckBox('Torso position')
        self.prefs_parabolas = QCheckBox('Calculated ball arcs')
        self.prefs_throwlabels = QCheckBox('Throw number labels')
        self.prefs_ideal_throws = QCheckBox(
                    'Ideal throw points and angles')

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
        controls_layout.addWidget(self.prefs_torso)
        controls_layout.addWidget(self.prefs_parabolas)
        controls_layout.addWidget(self.prefs_throwlabels)
        controls_layout.addWidget(self.prefs_ideal_throws)
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

    def makeStatsWidget(self):
        """
        A chart depicting descriptive information for a given run. This uses
        the custom widget HEStatsChart to do the display.
        """
        stats_layout = QVBoxLayout()
        layout_runselect = QHBoxLayout()
        layout_runselect.setAlignment(Qt.AlignLeft)
        layout_runselect.addWidget(QLabel('Run:'))
        self.stats_run = QComboBox()
        self.stats_run.setEditable(False)
        self.stats_run.currentIndexChanged.connect(self.statsRunChanged)
        layout_runselect.addWidget(self.stats_run)
        stats_layout.addLayout(layout_runselect)
        self.chartWidget = HEStatsChart(self)
        self.chartWidget.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        stats_layout.addWidget(self.chartWidget)
        stats_widget = QWidget()
        stats_widget.setLayout(stats_layout)
        return stats_widget

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
               '<p>&copy; 2018 Jack Boyce (jboyce@gmail.com)</p>'
               '<p>&nbsp;</p>'
               '<p>This software processes video files given to it and tracks '
               'objects moving through the air in parabolic trajectories. '
               'Currently it detects balls only. For best results use video '
               'with four or more objects, a high frame rate (60+ fps), '
               'minimum motion blur, and good brightness separation between '
               'the balls and background.</p>'
               '<p>Drop a video file onto the <b>Videos:</b> box to begin!</p>'
               '<p>Useful keyboard shortcuts when viewing video:</p>'
               '<ul>'
               '<li>space: toggle play/pause</li>'
               '<li>arrow keys: step forward/backward by one frame (hold to '
               'continue cueing)</li>'
               '<li>z, x: step backward/forward by one throw</li>'
               '<li>a, s: step backward/forward by one run</li>'
               '</ul>'
               '<p>&nbsp;</p>'
               '<p><small>All portions of this software written by Jack Boyce '
               'are provided under the MIT License. Other software '
               'distributed as part of Hawkeye is done so under the terms of '
               'their respective non-commercial licenses: '
               'OpenCV version 3.4.1 (3-clause BSD License), '
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

    def startWorker(self):
        """
        Starts up worker object on a separate thread, to handle the
        lengthy processing operations without blocking the UI.

        For data consistency all communication with the worker is done
        via Qt signals and slots.
        """
        self._worker = worker = HEWorker(app)
        self._thread = thread = QThread()
        worker.moveToThread(thread)

        self.sig_new_work.connect(worker.on_new_work)
        self.sig_new_prefs.connect(worker.on_new_prefs)
        self.sig_worker_quit.connect(worker.on_app_quit)
        worker.sig_progress.connect(self.on_worker_step)
        worker.sig_output.connect(self.on_worker_output)
        worker.sig_error.connect(self.on_worker_error)
        worker.sig_done.connect(self.on_worker_done)
        worker.sig_quit.connect(thread.quit)
        worker.sig_quit.connect(app.quit)
        thread.started.connect(worker.work)

        thread.start()
        self.sig_new_prefs.emit(self.prefs)
        self.worker_queue_length = 0

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
            if item._filepath == file_id:
                item._processing_step = step
                item._processing_steps_total = stepstotal
                if self.currentVideoItem is item:
                    self.progressBar.setValue(step)
                    self.progressBar.setMaximum(stepstotal)
                break

    @Slot(str, str)
    def on_worker_output(self, file_id: str, output: str):
        """
        Signaled by worker when there is output from processing.
        """
        for i in range(0, self.videoList.count()):
            item = self.videoList.item(i)
            if item is None:
                continue
            if item._filepath == file_id:
                item._output += output
                if self.currentVideoItem is item:
                    self.outputWidget.moveCursor(QTextCursor.End)
                    self.outputWidget.insertPlainText(output)
                    self.outputWidget.moveCursor(QTextCursor.End)
                break

    @Slot(str, str)
    def on_worker_error(self, file_id: str, errormsg: str):
        """
        Signaled by worker when there is an error
        """
        for i in range(0, self.videoList.count()):
            item = self.videoList.item(i)
            if item is None:
                continue
            if item._filepath == file_id:
                self.worker_queue_length -= 1
                item._notes = None
                item._doneprocessing = True
                item.setForeground(item._foreground)
                if self.currentVideoItem is item:
                    self.build_view_list(item)
                    self.progressBar.hide()
                break

    @Slot(str, dict, dict, int)
    def on_worker_done(self, file_id: str, notes: dict, fileinfo: dict,
                       resolution: int):
        """
        Signaled by worker when processing of a video is completed.
        """
        for i in range(self.videoList.count()):
            item = self.videoList.item(i)
            if item is None:
                continue
            if item._filepath == file_id:
                self.worker_queue_length -= 1
                item._videopath = fileinfo['displayvid_path']
                item._videoresolution = resolution
                item._csvpath = fileinfo['csvfile_path']
                item._notes = notes
                item._doneprocessing = True
                # see note in HEVideoList.addVideo():
                item._graphicsvideoitem = QGraphicsVideoItem()
                item._graphicsscene = QGraphicsScene(self.view)
                item._graphicsscene.addItem(item._graphicsvideoitem)
                item._graphicsvideoitem.nativeSizeChanged.connect(
                        self.videoNativeSizeChanged)
                item.setForeground(item._foreground)

                if item is self.currentVideoItem:
                    wantpaused = True
                    if self.views_stackedWidget.currentIndex() == 0:
                        # movie is currently being viewed, retain current pause
                        # state
                        wantpaused = (self.mediaPlayer.state() !=
                                      QMediaPlayer.PlayingState)

                    self.syncPlayer()
                    # time.sleep(0.5)
                    self.mediaPlayer.setPosition(
                            self.currentVideoItem._position)
                    if wantpaused:
                        self.pauseMovie()

                    self.build_view_list(item)
                    self.progressBar.hide()
                    self.views_stackedWidget.setCurrentIndex(0)

                break

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
        self.currentVideoItem = items[0]

        # self.pauseMovie()
        self.build_view_list(self.currentVideoItem)
        self.syncPlayer()

        if self.currentVideoItem._doneprocessing:
            # switch to video view, paused and cued to where we left off
            # last time
            self.mediaPlayer.setPosition(self.currentVideoItem._position)
            self.pauseMovie()
            self.views_stackedWidget.setCurrentIndex(0)
        else:
            # auto-select 'scanner output' view
            for i in range(self.viewList.count()):
                viewitem = self.viewList.item(i)
                if viewitem._type == 'output':
                    viewitem.setSelected(True)
                    break

    def build_view_list(self, videoitem):
        """
        Construct the set of views to make available for a given video. This
        populates the View list in the UI.
        """
        notes = videoitem._notes
        self.viewList.clear()

        if not videoitem._doneprocessing:
            headeritem = QListWidgetItem('')
            headeritem._type = 'video'
            headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
            header = QLabel('Video')
            self.viewList.addItem(headeritem)
            self.viewList.setItemWidget(headeritem, header)

        if videoitem._doneprocessing and notes is not None:
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
                    startframe, _ = notes['run'][i]['frame range']
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
            headeritem._type = 'stats'
            headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
            header = QLabel('Stats')
            self.viewList.addItem(headeritem)
            self.viewList.setItemWidget(headeritem, header)

            headeritem = QListWidgetItem('')
            headeritem._type = 'data'
            headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
            header = QLabel('Run data')
            self.viewList.addItem(headeritem)
            self.viewList.setItemWidget(headeritem, header)

        headeritem = QListWidgetItem('')
        headeritem._type = 'output'
        headeritem.setFlags(headeritem.flags() | Qt.ItemIsSelectable)
        header = QLabel('Scanner output')
        self.viewList.addItem(headeritem)
        self.viewList.setItemWidget(headeritem, header)

        if not videoitem._doneprocessing:
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
            self.mediaPlayer.setPosition(self.currentVideoItem._position)
            self.pauseMovie()
        elif item._type == 'output':
            self.pauseMovie()
            self.outputWidget.setPlainText(self.currentVideoItem._output)
            if self.currentVideoItem._doneprocessing:
                self.progressBar.hide()
            else:
                self.progressBar.setValue(
                        self.currentVideoItem._processing_step)
                self.progressBar.setMaximum(
                        self.currentVideoItem._processing_steps_total)
                self.progressBar.show()
            self.views_stackedWidget.setCurrentIndex(1)
        elif item._type == 'stats':
            notes = self.currentVideoItem._notes
            if notes is not None:
                self.fillStatsView(notes)
                self.views_stackedWidget.setCurrentIndex(2)
        elif item._type == 'data':
            notes = self.currentVideoItem._notes
            if notes is not None:
                self.fillDataView(notes)
                self.views_stackedWidget.setCurrentIndex(3)
        elif item._type == 'run':
            self.views_stackedWidget.setCurrentIndex(0)
            self.playMovie()
            self.setFramenum(item._startframe)
        else:
            # shouldn't ever get here
            pass

    def fillStatsView(self, notes):
        """
        Fill in the Stats view when the user wants to view it.
        """
        self.stats_run.clear()
        if notes is None or notes['runs'] < 1:
            return
        for run_num in range(notes['runs']):
            self.stats_run.insertItem(run_num, str(run_num + 1))
        self.stats_run.setCurrentText('1')

    @Slot(int)
    def statsRunChanged(self, index):
        """
        Called when a new run is selected in the Stats view combo box.
        """
        if self.currentVideoItem is None:
            return
        notes = self.currentVideoItem._notes
        if notes is not None:
            self.chartWidget.setRunDict(notes['run'][index])

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
                throw_hand = 'R' if arc.hand_throw == 'right' else 'L'
                catch_t = arc.f_catch / notes['fps'] - start_t
                catch_x, _ = arc.get_position(arc.f_catch, notes)
                catch_x *= notes['cm_per_pixel']
                catch_hand = 'R' if arc.hand_catch == 'right' else 'L'
                hands = throw_hand + '->' + catch_hand

                throw_vx = arc.b
                throw_vy = -2 * arc.e * (arc.f_throw - arc.f_peak)
                angle = atan(throw_vx / throw_vy) * 180 / pi
                height = 0.125 * 980.7 * (catch_t - throw_t)**2

                try:
                    x, _, w, _ = notes['body'][round(arc.f_peak)]
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

        csvfile_path = self.currentVideoItem._csvpath
        filename = QFileDialog.getSaveFileName(self, 'Save CSV File',
                                               csvfile_path)
        if filename != '':
            filepath = os.path.abspath(filename[0])
            with open(filepath, 'w') as f:
                f.write(output)

    def closeEvent(self, e):
        """
        Called when the user clicks the close icon in the window corner.
        This overrides QMainWindow.closeEvent()
        """
        self.exitCall()

    def openFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Video',
                                                  QDir.homePath())
        if filename != '':
            filepath = os.path.abspath(filename)
            lastitem = self.videoList.addVideo(filepath)
            if lastitem is not None:
                lastitem.setSelected(True)

    def syncPlayer(self):
        """
        This should be called whenever the currently-viewed movie changes. It
        syncs the media player to the new video file and adjusts other UI
        elements as needed.
        """
        if self.currentVideoItem is None:
            return
        videopath = self.currentVideoItem._videopath
        if videopath is None:
            # no converted video yet: default to original video file
            videopath = self.currentVideoItem._filepath
        print('syncing player to video file {}'.format(videopath))

        self.item = self.currentVideoItem._graphicsvideoitem
        self.scene = self.currentVideoItem._graphicsscene
        self.view.setScene(self.scene)
        self.mediaPlayer.setVideoOutput(self.item)

        self.mediaPlayer.pause()
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(videopath)))
        self.mediaPlayer.play()

        self.playButton.setEnabled(True)
        self.backButton.setEnabled(False)
        self.forwardButton.setEnabled(False)

    @Slot()
    def exitCall(self):
        """
        Called when the user selects Quit in the menu, or clicks the close
        box in the window's corner.

        All we do is tell the worker thread to abort: It will clean up and do
        the actual quit.
        """
        self.sig_worker_quit.emit()

    @Slot()
    def togglePlay(self):
        """
        Called when the user clicks the play/pause button in the UI.
        """
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.pauseMovie()
        else:
            self.playMovie()

    def pauseMovie(self):
        """
        Utility function to pause the movie if it's playing. Be careful to stop
        on a clean frame boundary, otherwise HEVideoView.paintEvent() can get
        off-by-one errors in the frame # when drawing overlays on top of the
        paused video.
        """
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            if self.currentVideoItem._notes is not None:
                position = self.mediaPlayer.position()
                framenum = self.framenumForPosition(position)
                self.setFramenum(framenum)
                # can't step frame-by-frame if we don't know fps
                self.backButton.setEnabled(True)
                self.forwardButton.setEnabled(True)

    def playMovie(self):
        """
        Utility function to un-pause the video.
        """
        if self.mediaPlayer.state() != QMediaPlayer.PlayingState:
            self.mediaPlayer.play()
            self.backButton.setEnabled(False)
            self.forwardButton.setEnabled(False)

    @Slot()
    def showPrefs(self):
        """
        Called when the user clicks the preferences icon in the UI.
        """
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.pauseMovie()
        self.prefs_markers.setChecked(self.prefs['markers'])
        self.prefs_torso.setChecked(self.prefs['torso'])
        self.prefs_parabolas.setChecked(self.prefs['parabolas'])
        self.prefs_throwlabels.setChecked(self.prefs['throw_labels'])
        self.prefs_ideal_throws.setChecked(self.prefs['ideal_throws'])
        self.prefs_resolution.setCurrentText(self.prefs['resolution'])
        self.player_stackedWidget.setCurrentIndex(1)

    @Slot()
    def setPrefs(self):
        """
        Called when the user clicks "Accept" in the preferences panel.
        """
        old_resolution = self.prefs['resolution']
        self.prefs['markers'] = self.prefs_markers.isChecked()
        self.prefs['torso'] = self.prefs_torso.isChecked()
        self.prefs['parabolas'] = self.prefs_parabolas.isChecked()
        self.prefs['throw_labels'] = self.prefs_throwlabels.isChecked()
        self.prefs['ideal_throws'] = self.prefs_ideal_throws.isChecked()
        self.prefs['resolution'] = self.prefs_resolution.currentText()
        self.sig_new_prefs.emit(self.prefs)
        self.player_stackedWidget.setCurrentIndex(0)

        # If display resolution changed, reprocess all videos through worker
        # to create the needed video files. Do the current video first.
        if self.prefs['resolution'] != old_resolution:
            if self.currentVideoItem is not None:
                self.sig_new_work.emit(self.currentVideoItem._filepath)
                self.worker_queue_length += 1

                # auto-select 'scanner output' view
                for i in range(self.viewList.count()):
                    viewitem = self.viewList.item(i)
                    if viewitem._type == 'output':
                        viewitem.setSelected(True)
                        break

            for i in range(self.videoList.count()):
                item = self.videoList.item(i)
                if item is not None and item is not self.currentVideoItem:
                    self.sig_new_work.emit(item._filepath)
                    self.worker_queue_length += 1

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
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.pauseMovie()
        self.views_stackedWidget.setCurrentIndex(4)

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
            itemPolygon = self.item.mapFromScene(scenePolygon)
            itemRect = itemPolygon.boundingRect()
            if itemRect.contains(self.item.boundingRect()):
                self.view.fitInView(self.item.boundingRect(),
                                    Qt.KeepAspectRatio)
                self.zoomOutButton.setEnabled(False)
                self.view.videosnappedtoframe = True
                self.view.setDragMode(QGraphicsView.NoDrag)

    def framenum(self):
        if self.currentVideoItem._notes is None:
            return None
        pos = self.mediaPlayer.position()
        framenum = floor(pos * self.currentVideoItem._notes['fps'] / 1000)
        return framenum

    def setFramenum(self, framenum):
        if self.currentVideoItem._notes is None:
            return
        fps = self.currentVideoItem._notes['fps']
        newpos = ceil(framenum * 1000 / fps) + floor(0.5 * 1000 / fps)
        self.mediaPlayer.setPosition(newpos)

    def framenumForPosition(self, position):
        if self.currentVideoItem._notes is None:
            return None
        return floor(position * self.currentVideoItem._notes['fps'] / 1000)

    @Slot()
    def stepBack(self):
        """
        Called when the user clicks the 'back one frame' button in the UI.
        """
        if self.currentVideoItem._notes is None:
            return
        self.pauseMovie()
        newframenum = self.framenum() - 1
        self.setFramenum(newframenum)

    @Slot()
    def stepForward(self):
        """
        Called when the user clicks the 'forward one frame' button in the UI.
        """
        if self.currentVideoItem._notes is None:
            return
        self.pauseMovie()
        newframenum = self.framenum() + 1
        self.setFramenum(newframenum)

    def mediaStateChanged(self, state):
        """
        Signaled by the QMediaPlayer when the state of the media changes from
        paused to playing, and vice versa.
        """
        """
        state = self.mediaPlayer.state()
        status = self.mediaPlayer.mediaStatus()
        print(f'media state = {state}, status = {status}')
        """
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def videoNativeSizeChanged(self, size):
        """
        Signaled by the QMediaPlayer when the video size changes.
        """
        self.item.setSize(size)
        self.view.fitInView(self.item.boundingRect(), Qt.KeepAspectRatio)

        self.zoomInButton.setEnabled(True)
        self.zoomOutButton.setEnabled(False)
        self.view.videosnappedtoframe = True

    def setPosition(self, position):
        """
        Signaled when user interacts with the position slider.

        This is NOT called when the position slider moves as a result of
        playing the movie. See positionChanged().
        """
        if self.currentVideoItem._notes is not None:
            framenum = self.framenumForPosition(position)
            self.setFramenum(framenum)
        else:
            self.mediaPlayer.setPosition(position)

    def positionChanged(self, position):
        """
        Signaled by the QMediaPlayer when position in the movie changes.

        We block signals since this is only called by the media player when
        position changes, not when the user interacts with the slider. We want
        to update the slider position without causing setPosition() to trigger.
        """
        prev = self.positionSlider.blockSignals(True)
        self.positionSlider.setValue(position)
        self.positionSlider.blockSignals(prev)

    def durationChanged(self, duration):
        """
        Signaled by the QMediaPlayer when the movie duration changes.
        """
        if duration > 0:
            self.positionSlider.setRange(0, duration)
            if self.currentVideoItem is not None:
                self.currentVideoItem._duration = duration

    @Slot(int)
    def on_rate_change(self, index):
        """
        Called when the user selects a playback rate on the dropdown menu.
        """
        self.mediaPlayer.setPlaybackRate(float(
                    self.playbackRate.currentText()))

    @Slot()
    def handlePlayerError(self):
        self.playButton.setEnabled(False)
        err = self.mediaPlayer.errorString()
        code = self.mediaPlayer.error()
        self.playerErrorLabel.setText(f'Error: {err} (code {code})')
        """
        state = self.mediaPlayer.state()
        status = self.mediaPlayer.mediaStatus()
        print('player error, media state = {}, status = {}'.format(state,
                                                                   status))
        """

    def keyPressEvent(self, e):
        """
        We have to work around the fact that neither Qt nor Python can give
        us the instantanous state of the keyboard, so we use key repeats to
        give us the desired function of the arrow keys. Do all of the
        repeated frame advancing in HEVideoView.paintEvent() so we don't get
        frame skips.

        keycodes at http://doc.qt.io/qt-5/qt.html#Key-enum
        """
        if self.views_stackedWidget.currentIndex() != 0:
            super().keyPressEvent(e)
            return

        key = e.key()
        framenum = self.framenum()
        notes = self.currentVideoItem._notes
        viewitem = self.currentViewItem

        if key == Qt.Key_Space:
            self.togglePlay()
        elif key == Qt.Key_Right:
            # advance movie by one frame
            self.playBackwardUntil = None
            if self.playForwardUntil is None:
                # not already playing forward
                self.playForwardUntil = framenum + 1
                self.pauseMovie()
                self.stepForward()
            else:
                """
                Set target two frames ahead so that HEVideoView.paintEvent()
                won't stop the forward playing on the next time it executes.
                This gives smooth advancement whether the key repeat rate is
                faster or slower than the frame refresh rate, at the cost of
                overshooting by one frame.
                """
                self.playForwardUntil = framenum + 2
        elif key == Qt.Key_Left:
            # step back one frame
            self.playForwardUntil = None
            if self.playBackwardUntil is None:
                self.playBackwardUntil = framenum - 1
                self.pauseMovie()
                self.stepBack()
            else:
                self.playBackwardUntil = framenum - 2
        elif key == Qt.Key_X:
            # play forward until next throw in run
            if notes is None:
                return
            if (viewitem is not None and viewitem._type == 'run'):
                runindex = viewitem._runindex
                throwarcs = notes['run'][runindex]['throw']
            else:
                throwarcs = notes['arcs']

            throwframes = sorted([arc.f_throw for arc in throwarcs
                                 if arc.f_throw > framenum + 1])
            if len(throwframes) == 0:
                return
            self.playForwardUntil = max(round(throwframes[0]), framenum + 1)
            self.playBackwardUntil = None
            self.pauseMovie()
            self.stepForward()
        elif key == Qt.Key_Z:
            # play backward until previous throw in run
            if notes is None:
                return
            if (self.currentViewItem is not None and
                    self.currentViewItem._type == 'run'):
                runindex = self.currentViewItem._runindex
                throwarcs = notes['run'][runindex]['throw']
            else:
                throwarcs = notes['arcs']

            throwframes = sorted([arc.f_throw for arc in throwarcs
                                 if arc.f_throw < framenum - 1], reverse=True)
            if len(throwframes) == 0:
                return
            self.playForwardUntil = None
            self.playBackwardUntil = min(round(throwframes[0]), framenum - 1)
            self.pauseMovie()
            self.stepBack()
        elif key == Qt.Key_S:
            # go to next run
            if (viewitem is not None and viewitem._type == 'run'):
                row = self.viewList.row(viewitem)
                nextitem = self.viewList.item(row + 1)
                if nextitem is not None and nextitem._type == 'run':
                    prev = self.viewList.blockSignals(True)
                    self.viewList.setCurrentRow(row + 1)
                    self.viewList.blockSignals(prev)
                    self.currentViewItem = nextitem
                    self.setFramenum(nextitem._startframe)
                    self.playForwardUntil = None
                    self.playBackwardUntil = None
        elif key == Qt.Key_A:
            # go to previous run, or start of current run if we're more than
            # one second into playback
            if (viewitem is not None and viewitem._type == 'run'):
                if framenum > (viewitem._startframe + notes['fps']):
                    self.setFramenum(viewitem._startframe)
                    self.playForwardUntil = None
                    self.playBackwardUntil = None
                else:
                    row = self.viewList.row(viewitem)
                    nextitem = self.viewList.item(row - 1)
                    if nextitem is not None and nextitem._type == 'run':
                        prev = self.viewList.blockSignals(True)
                        self.viewList.setCurrentRow(row - 1)
                        self.viewList.blockSignals(prev)
                        self.currentViewItem = nextitem
                        self.setFramenum(nextitem._startframe)
                        self.playForwardUntil = None
                        self.playBackwardUntil = None
        else:
            super().keyPressEvent(e)

    def sizeHint(self):
        return QSize(850, 600)

    def event(self, e):
        if debug:
            print('event type {}'.format(e.type()))
        return super().event(e)

# -----------------------------------------------------------------------------


class HEVideoView(QGraphicsView):
    """
    Subclass of QGraphicsView so we can override the paintEvent() method.
    This lets us display frames from a video with graphics overlaid on top.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.window = parent

        self.videosnappedtoframe = False
        if debug:
            self.counter = 0

    def paintEvent(self, e):
        # draw the video frame
        t_before = time.time()
        super().paintEvent(e)
        if debug:
            self.counter += 1
            if self.counter >= 20:
                self.counter = 0
                t_after = time.time()
                pending = app.hasPendingEvents()
                print('paintEvent duration {}, events pending = {}'.format(
                        t_after - t_before, pending))

        if self.window is None:
            return
        videoitem = self.window.currentVideoItem
        if videoitem is None:
            return
        notes = videoitem._notes
        prefs = self.window.prefs

        # record current position in movie, so we can save our place if
        # we switch between movies
        pos = self.window.mediaPlayer.position()
        videoitem._position = pos

        # decide if we should draw anything else
        if notes is None:
            return
        moviestatus = self.window.mediaPlayer.mediaStatus()
        if (moviestatus != QMediaPlayer.BufferedMedia and
                moviestatus != QMediaPlayer.BufferingMedia and
                moviestatus != QMediaPlayer.EndOfMedia):
            return

        framenum = self.window.framenum()
        framenum_max = self.window.framenumForPosition(
                self.window.mediaPlayer.duration())

        # the following is a hack to solve the way QMediaPlayer on Windows
        # reports positions within videos, so we don't get an off-by-one error
        if platform.system() == 'Windows':
            notes_framenum = framenum + 1
        else:
            notes_framenum = framenum

        # check if we're at the beginning or end of the movie
        if framenum == 0:
            self.window.backButton.setEnabled(False)
        elif (framenum >= (framenum_max - 1) or
              moviestatus == QMediaPlayer.EndOfMedia):
            self.window.mediaPlayer.play()
            self.window.mediaPlayer.pause()
            self.window.setFramenum(framenum_max - 1)
            self.window.backButton.setEnabled(True)
            self.window.forwardButton.setEnabled(False)
        else:
            paused = (self.window.mediaPlayer.state() !=
                      QMediaPlayer.PlayingState)
            self.window.backButton.setEnabled(paused)
            self.window.forwardButton.setEnabled(paused)

        if (self.window.playForwardUntil is not None and
                framenum >= self.window.playForwardUntil):
            self.window.pauseMovie()
            self.window.setFramenum(self.window.playForwardUntil)
            self.window.playForwardUntil = None
        if (self.window.playBackwardUntil is not None and
                framenum <= self.window.playBackwardUntil):
            self.window.pauseMovie()
            self.window.setFramenum(self.window.playBackwardUntil)
            self.window.playBackwardUntil = None

        # keep selected run in viewlist synchronized with position in movie
        vl = self.window.viewList
        for i in range(vl.count()):
            viewitem = vl.item(i)
            if (viewitem._type == 'run' and
                    (viewitem._startframe <= framenum < viewitem._endframe or
                     (viewitem._runindex == 0 and
                      framenum < viewitem._startframe))):
                if not viewitem.isSelected():
                    prev = vl.blockSignals(True)
                    vl.setCurrentRow(i)
                    self.window.currentViewItem = viewitem
                    vl.blockSignals(prev)
                    break

        vp = self.viewport()
        painter = QPainter(vp)

        if videoitem._videoresolution == 0:
            def mapToDisplayVideo(x, y):
                return x, y
        else:
            orig_height = notes['frame_height']
            orig_width = notes['frame_width']
            display_height = videoitem._videoresolution
            display_width = round(orig_width * display_height / orig_height)
            if display_width % 2 == 1:
                display_width += 1

            def mapToDisplayVideo(x, y):
                new_x = x * display_width / orig_width
                new_y = y * display_height / orig_height
                return new_x, new_y

        # draw box around juggler torso
        if prefs['torso'] and notes_framenum in notes['body']:
            x, y, w, h = notes['body'][notes_framenum]
            dUL_x, dUL_y = mapToDisplayVideo(x, y)
            dLR_x, dLR_y = mapToDisplayVideo(x + w, y + h)
            # upper-left and lower-right points respectively:
            bodyUL_x, bodyUL_y = self.mapToView(dUL_x, dUL_y)
            bodyLR_x, bodyLR_y = self.mapToView(dLR_x, dLR_y)
            painter.setPen(Qt.blue)
            painter.setBrush(Qt.NoBrush)
            painter.setOpacity(1.0)
            painter.drawRect(bodyUL_x, bodyUL_y,
                             bodyLR_x - bodyUL_x, bodyLR_y - bodyUL_y)

        # draw object detections
        if self.window.prefs['markers'] and notes_framenum in notes['meas']:
            for tag in notes['meas'][notes_framenum]:
                color = Qt.green if tag.arc is not None else Qt.red
                painter.setPen(color)
                painter.setBrush(Qt.NoBrush)
                painter.setOpacity(1.0)
                dcenter_x, dcenter_y = mapToDisplayVideo(tag.x, tag.y)
                center_x, center_y = self.mapToView(dcenter_x, dcenter_y)
                dright_x, dright_y = mapToDisplayVideo(tag.x + tag.radius,
                                                       tag.y)
                right_x, _ = self.mapToView(dright_x, dright_y)
                radius = right_x - center_x
                painter.drawEllipse(QPoint(center_x, center_y), radius, radius)

        points_per_parabola = 20
        for arc in notes['arcs']:
            start, end = round(arc.f_throw), round(arc.f_catch)
            if start <= framenum <= end:
                # draw arc parabola if it's visible on this frame
                if prefs['parabolas']:
                    path = QPainterPath()
                    for i in range(points_per_parabola):
                        f_point = (arc.f_throw + i *
                                   (arc.f_catch - arc.f_throw) /
                                   (points_per_parabola - 1))
                        x, y = arc.get_position(f_point, notes)
                        dx, dy = mapToDisplayVideo(x, y)
                        arc_x, arc_y = self.mapToView(dx, dy)
                        if i == 0:
                            path.moveTo(arc_x, arc_y)
                        else:
                            path.lineTo(arc_x, arc_y)
                    color = Qt.red if framenum == start else Qt.green
                    painter.setPen(color)
                    painter.setBrush(Qt.NoBrush)
                    painter.setOpacity((end - framenum) / (end - start))
                    painter.drawPath(path)

                """
                # draw marker of arc position
                x, y = arc.get_position(notes_framenum, notes)
                arc_x, arc_y = self.mapToView(x, y)
                arc_has_tag = any(
                        arc.get_distance_from_tag(tag, notes) <
                        notes['scanner_params']['max_distance_pixels']
                        for tag in arc.tags if tag.frame == notes_framenum)
                color = Qt.green if arc_has_tag else Qt.red
                painter.setPen(color)
                painter.setBrush(color)
                painter.setOpacity(1.0)
                painter.drawEllipse(QPoint(arc_x, arc_y), 2, 2)
                """

                if framenum == start:
                    # special things to draw when we're at the exact
                    # starting frame of a throw

                    # draw ideal throw arc
                    if prefs['ideal_throws'] and arc.ideal is not None:
                        ideal = arc.ideal
                        path = QPainterPath()
                        for i in range(points_per_parabola):
                            f_point = ideal.f_throw + i * (
                                    (ideal.f_catch - ideal.f_throw) /
                                    (points_per_parabola - 1))
                            x, y = ideal.get_position(f_point, notes)
                            dx, dy = mapToDisplayVideo(x, y)
                            ideal_x, ideal_y = self.mapToView(dx, dy)
                            if i == 0:
                                path.moveTo(ideal_x, ideal_y)
                            else:
                                path.lineTo(ideal_x, ideal_y)
                        painter.setPen(Qt.white)
                        painter.setBrush(Qt.NoBrush)
                        painter.setOpacity(1.0)
                        painter.drawPath(path)

                        x, y = ideal.get_position(ideal.f_throw, notes)
                        dx, dy = mapToDisplayVideo(x, y)
                        ideal_x, ideal_y = self.mapToView(dx, dy)
                        painter.setPen(Qt.white)
                        painter.setBrush(Qt.white)
                        painter.setOpacity(1.0)
                        painter.drawEllipse(QPoint(ideal_x, ideal_y), 2, 2)

                    # draw vectors of closest approach with other arcs
                    if prefs['parabolas'] and arc.close_arcs is not None:
                        for arc2_throw_id, frame, _ in arc.close_arcs[:5]:
                            arc2 = next((t for t in notes['arcs']
                                         if t.throw_id == arc2_throw_id
                                         and t.run_id == arc.run_id), None)
                            if arc2 is None:
                                continue
                            x1, y1 = arc.get_position(frame, notes)
                            dx1, dy1 = mapToDisplayVideo(x1, y1)
                            x1, y1 = self.mapToView(dx1, dy1)
                            x2, y2 = arc2.get_position(frame, notes)
                            dx2, dy2 = mapToDisplayVideo(x2, y2)
                            x2, y2 = self.mapToView(dx2, dy2)

                            color = (Qt.red if arc2.f_throw < arc.f_throw
                                     else Qt.blue)
                            painter.setPen(color)
                            painter.setBrush(color)
                            painter.setOpacity(1.0)
                            painter.drawEllipse(QPoint(x1, y1), 2, 2)
                            painter.setBrush(Qt.NoBrush)
                            painter.drawEllipse(QPoint(x2, y2), 2, 2)
                            painter.drawLine(round(x1), round(y1),
                                             round(x2), round(y2))

                # draw throw number next to arc position
                if prefs['throw_labels']:
                    try:
                        label = format(arc.throw_id, ' 4d')
                        x, y = arc.get_position(notes_framenum, notes)
                        dx, dy = mapToDisplayVideo(x, y)
                        arc_x, arc_y = self.mapToView(dx, dy)
                        painter.setOpacity(1.0)
                        painter.fillRect(arc_x+15, arc_y-5, 25, 9, Qt.black)
                        font = painter.font()
                        font.setFamily('Courier')
                        font.setPixelSize(9)
                        painter.setFont(font)
                        painter.setPen(Qt.white)
                        painter.drawText(arc_x+15, arc_y+3, label)
                    except AttributeError:
                        pass

        # draw frame number in lower left corner
        digits = 1 if framenum <= 1 else 1 + floor(log10(framenum))
        digits_max = 1 if framenum_max <= 1 else 1 + floor(log10(framenum_max))
        movie_lower_left_x, movie_lower_left_y = self.mapToView(
                0.0, self.window.item.size().height())
        lower_left_x = max(0, movie_lower_left_x)
        lower_left_y = min(vp.size().height(), movie_lower_left_y)
        painter.setOpacity(1.0)
        painter.fillRect(lower_left_x, lower_left_y - 20,
                         10 + 12 * digits_max, 20, Qt.black)
        font = painter.font()
        font.setFamily('Courier')
        font.setPixelSize(20)
        painter.setFont(font)
        painter.setPen(Qt.white)
        painter.drawText(lower_left_x + 5 + 12 * (digits_max - digits),
                         lower_left_y - 3, str(framenum))

        if self.window.playForwardUntil is not None:
            self.window.stepForward()
        elif self.window.playBackwardUntil is not None:
            self.window.stepBack()

    def mapToView(self, x, y):
        """
        map coordinate from movie coordinates (pixels) to view coordinates
        """
        movie_coord = QPointF(x, y)
        scene_coord = self.window.item.mapToScene(movie_coord)
        view_coord = self.mapFromScene(scene_coord)
        return view_coord.x(), view_coord.y()

    def resizeEvent(self, e):
        # print('resizeEvent at time {}'.format(time.time()))
        if self.window is not None:
            if self.videosnappedtoframe:
                self.fitInView(self.scene().itemsBoundingRect(),
                               Qt.KeepAspectRatio)
            else:
                # see if the entire movie frame is now visible; if so then
                # snap to frame
                portRect = self.viewport().rect()
                scenePolygon = self.mapToScene(portRect)
                itemPolygon = self.window.item.mapFromScene(scenePolygon)
                itemRect = itemPolygon.boundingRect()
                if itemRect.contains(self.window.item.boundingRect()):
                    self.fitInView(self.window.item.boundingRect(),
                                   Qt.KeepAspectRatio)
                    self.window.zoomOutButton.setEnabled(False)
                    self.videosnappedtoframe = True
                    self.setDragMode(QGraphicsView.NoDrag)

        super().resizeEvent(e)

    def sizeHint(self):
        return QSize(640, 480)

# -----------------------------------------------------------------------------


class HEVideoList(QListWidget):
    """
    Subclass of standard QListWidget that implements drag-and-drop of files
    onto the list.
    """
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.window = parent
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        selectitem = None
        workerfree = (self.window.worker_queue_length == 0)
        for url in e.mimeData().urls():
            filepath = os.path.abspath(url.toLocalFile())
            item = self.addVideo(filepath)
            if workerfree and selectitem is None:
                selectitem = item
        if selectitem is not None:
            selectitem.setSelected(True)

    def addVideo(self, filepath):
        if not os.path.isfile(filepath):
            return None

        basename = os.path.basename(filepath)
        item = QListWidgetItem(basename)
        item._filepath = filepath
        item._notes = None
        item._videopath = filepath
        item._videoresolution = 0
        item._foreground = item.foreground()
        item._output = ''
        item._doneprocessing = False
        item._position = 0
        item._processing_step = 0
        item._processing_steps_total = 0
        """
        QGraphicsVideoItem and QGraphicsScene have scaling problems when
        the video changes. So we keep a separate per-video instance of each
        and swap them in when we change videos in the view. See syncPlayer().
        """
        item._graphicsvideoitem = QGraphicsVideoItem()
        item._graphicsscene = QGraphicsScene(self.window.view)
        item._graphicsscene.addItem(item._graphicsvideoitem)
        item._graphicsvideoitem.nativeSizeChanged.connect(
                self.window.videoNativeSizeChanged)
        item.setForeground(Qt.gray)
        item.setFlags(item.flags() | Qt.ItemIsSelectable)
        self.addItem(item)

        window.sig_new_work.emit(filepath)
        self.window.worker_queue_length += 1
        return item

    def sizeHint(self):
        return QSize(150, 100)

# -----------------------------------------------------------------------------


class HEViewList(QListWidget):
    """
    Subclass of standard QListWidget to show run-related info for a video. I
    wonder why QWidget doesn't have a setSizeHint() method.
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def sizeHint(self):
        return QSize(150, 100)

# -----------------------------------------------------------------------------


class HETableViewDelegate(QStyledItemDelegate):
    """
    Subclass of QStyledItemDelegate to draw horizontal lines between
    separate runs in our Data View QTableWidget.
    """
    def __init__(self, table_widget, boundary_indices):
        super().__init__()
        self.boundary_indices = boundary_indices
        self.pen = QPen(table_widget.gridStyle())

    def paint(self, painter, item, index):
        super().paint(painter, item, index)
        if index.row() in self.boundary_indices:
            oldPen = painter.pen()
            painter.setPen(self.pen)
            painter.drawLine(item.rect.topLeft(), item.rect.topRight())
            painter.setPen(oldPen)

# -----------------------------------------------------------------------------


class HEStatsChart(QWidget):
    """
    Widget that draws the statistics chart for a given run.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_dict = None

    def setRunDict(self, run_dict):
        self.run_dict = run_dict
        # print('set new run, throws = {}'.format(run_dict['throws']))

    def paintEvent(self, e):
        super().paintEvent(e)
        w, h = self.width(), self.height()
        xl, xu = 0.25 * w, 0.3 * w
        f = 0.02
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # white background
        pen = QPen(Qt.white)
        painter.setPen(pen)
        painter.setBrush(Qt.white)
        painter.drawRect(0, 0, w - 1, h - 1)

        # axes
        pen.setColor(Qt.lightGray)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(f * w, 0.25 * h, (1 - f) * w, 0.25 * h)
        painter.drawLine(f * w, 0.75 * h, (1 - f) * w, 0.75 * h)
        painter.drawLine(xl, (0.5 + f) * h, xl, (1 - f) * h)
        painter.drawLine(w - xl, (0.5 + f) * h, w - xl, (1 - f) * h)
        painter.drawLine(xu, f * h, xu, (0.5 - f) * h)
        painter.drawLine(w - xu, f * h, w - xu, (0.5 - f) * h)
        if self.run_dict is None:
            return

# -----------------------------------------------------------------------------


if __name__ == '__main__':
    profile = False
    debug = False

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = HEMainWindow()
    window.show()
    window.raise_()
    app.setActiveWindow(window)

    if profile:
        import cProfile

        cProfile.run('app.exec_()')
        sys.exit()
    else:
        sys.exit(app.exec_())
