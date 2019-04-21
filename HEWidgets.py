# HEWidgets.py
#
# Custom Qt widgets for the Hawkeye application.
#
# Copyright 2019 Jack Boyce (jboyce@gmail.com)

import os
import platform
from math import log10, floor

from PySide2.QtCore import QObject, Slot, QSize, Qt, QPoint, QPointF
from PySide2.QtGui import QPainter, QPainterPath, QPen
from PySide2.QtWidgets import (QWidget, QGraphicsView, QGraphicsScene,
                               QListWidget, QListWidgetItem, QAbstractItemView,
                               QStyledItemDelegate, QStyle)
from PySide2.QtMultimedia import QMediaPlayer
from PySide2.QtMultimediaWidgets import QGraphicsVideoItem


class HEVideoView(QGraphicsView):
    """
    Subclass of QGraphicsView so we can override the paintEvent() method.
    This lets us display frames from a video with graphics overlaid on top.
    """
    def __init__(self, main_window):
        super().__init__(parent=main_window)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.window = main_window
        self.videosnappedtoframe = True

    def do_playback_control(self, framenum):
        """
        Called from paintEvent().

        Adjust UI elements as needed during playback, and perform aspects of
        playback control that we want to be frame-accurate.

        There are two modes of video playback: Regular playback of the media
        (at the rate specified in the pulldown control), and "fast stepping"
        mode. Fast stepping is done by pausing regular playback and manually
        advancing forward or backward through successive frames with
        setFramenum(). This allows allows us to cue forward or backward to a
        precise frame number.
        """
        win = self.window
        vc = self.window.currentVideoItem.vc

        # record position in movie
        vc.position = vc.player.position()

        # update position of slider, and block signals so we don't trigger
        # a call to HEMainWindow.setPosition()
        prev = win.positionSlider.blockSignals(True)
        win.positionSlider.setValue(vc.position)
        win.positionSlider.blockSignals(prev)

        # check if we're at the beginning or end of the video and adjust
        # UI elements as appropriate
        if framenum == 0:
            win.backButton.setEnabled(False)
        elif (framenum >= vc.frames - 1):
            win.pauseMovie()
            if framenum != vc.frames - 1:
                win.setFramenum(vc.frames - 1)
            framenum = vc.frames - 1
            win.backButton.setEnabled(vc.has_played)
            win.forwardButton.setEnabled(False)
            win.playButton.setEnabled(False)
        else:
            can_step = (vc.player.state() != QMediaPlayer.PlayingState
                        and vc.has_played)
            win.backButton.setEnabled(can_step)
            win.forwardButton.setEnabled(can_step)
            win.playButton.setEnabled(True)

        # keep selected run in viewlist synchronized with position in video
        for i in range(win.viewList.count()):
            viewitem = win.viewList.item(i)
            if (viewitem._type == 'run' and
                    (viewitem._startframe <= framenum < viewitem._endframe or
                     (viewitem._runindex == 0 and
                      framenum < viewitem._startframe))):
                if not viewitem.isSelected():
                    prev = win.viewList.blockSignals(True)
                    win.viewList.setCurrentRow(i)
                    win.viewList.blockSignals(prev)
                    win.currentViewItem = viewitem
                    break

        # if we're in fast-stepping mode, ensure we respect the frame limits
        # and also check if we've hit the end of the range we're stepping
        # through.
        if win.stepForwardUntil is not None:
            win.stepForwardUntil = min(win.stepForwardUntil, vc.frames - 1)
            if framenum >= win.stepForwardUntil:
                if framenum > win.stepForwardUntil:
                    # shouldn't happen, but just in case
                    win.setFramenum(win.stepForwardUntil)
                win.stepForwardUntil = None
        if win.stepBackwardUntil is not None:
            win.stepBackwardUntil = max(win.stepBackwardUntil, 0)
            if framenum <= win.stepBackwardUntil:
                if framenum < win.stepBackwardUntil:
                    win.setFramenum(win.stepBackwardUntil)
                win.stepBackwardUntil = None

        # if we're in fast-stepping mode, advance a frame
        if win.stepForwardUntil is not None:
            win.stepForward()
        elif win.stepBackwardUntil is not None:
            win.stepBackward()

    def draw_overlays(self, painter, framenum):
        """
        Called from paintEvent().

        Draw any overlays on the video frame, for the current frame number.
        """
        if not self.window.currentVideoItem.vc.overlays:
            return
        notes = self.window.currentVideoItem.vc.notes
        if notes is None or notes['step'] < 5:
            return

        # the following is a hack to solve the way QMediaPlayer on Windows
        # reports positions within videos, so we don't get an off-by-one error
        if platform.system() == 'Windows':
            notes_framenum = framenum + 1
        else:
            notes_framenum = framenum

        prefs = self.window.prefs
        mapToDisplayVideo = self.window.currentVideoItem.vc.map

        # draw box around juggler torso
        if prefs['torso'] and notes_framenum in notes['body']:
            x, y, w, h, _ = notes['body'][notes_framenum]
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
        if prefs['markers'] and notes_framenum in notes['meas']:
            for tag in notes['meas'][notes_framenum]:
                if tag.arc is None:
                    color = Qt.red
                else:
                    color = (Qt.yellow if tag.arc.hand_throw == 'right'
                             else Qt.green)
                dcenter_x, dcenter_y = mapToDisplayVideo(tag.x, tag.y)
                center_x, center_y = self.mapToView(dcenter_x, dcenter_y)

                """
                # style = circles surrounding balls:
                dright_x, dright_y = mapToDisplayVideo(
                        tag.x + tag.radius, tag.y)
                right_x, _ = self.mapToView(dright_x, dright_y)
                radius = right_x - center_x
                painter.setBrush(Qt.NoBrush)
                """

                # style = filled circles inside balls:
                marker_radius_px = 1.5 / notes['cm_per_pixel']
                dright_x, dright_y = mapToDisplayVideo(
                        tag.x + marker_radius_px, tag.y)
                right_x, _ = self.mapToView(dright_x, dright_y)
                radius = max(right_x - center_x, 2.0)
                painter.setBrush(color)

                painter.setPen(color)
                painter.setOpacity(1.0)
                painter.drawEllipse(QPoint(center_x, center_y), radius, radius)

        points_per_parabola = 20

        for arc in notes['arcs']:
            # find arcs that are visible in this frame
            start, end = round(arc.f_throw), round(arc.f_catch)
            if framenum < start or framenum > end:
                continue

            # draw arc parabola
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
                color = Qt.yellow if arc.hand_throw == 'right' else Qt.green
                if framenum == start:
                    color = Qt.red
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
                # special things to draw when we're at the exact starting
                # frame of a throw

                """
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
                """

                # draw hand's carry from previous catch, if any
                if prefs['carries'] and arc.prev is not None:
                    xt, yt = arc.get_position(arc.f_throw, notes)
                    dxt, dyt = mapToDisplayVideo(xt, yt)
                    xt, yt = self.mapToView(dxt, dyt)
                    xc, yc = arc.prev.get_position(arc.prev.f_catch, notes)
                    dxc, dyc = mapToDisplayVideo(xc, yc)
                    xc, yc = self.mapToView(dxc, dyc)

                    painter.setPen(Qt.red)
                    painter.setBrush(Qt.red)
                    painter.setOpacity(1.0)
                    painter.drawEllipse(QPoint(xt, yt), 2, 2)
                    painter.drawEllipse(QPoint(xc, yc), 2, 2)
                    painter.drawLine(round(xt), round(yt),
                                     round(xc), round(yc))

                # draw ideal throw and catch points
                if prefs['ideal_points']:
                    # find centerline
                    x, y, w, h, _ = notes['body'][notes_framenum]
                    center_x_px = x + 0.5 * w      # pixel units
                    center_y_px = y + h

                    run_dict = notes['run'][arc.run_id - 1]
                    throw_offset_px = (run_dict['target throw point cm']
                                       / notes['cm_per_pixel'])
                    catch_offset_px = (run_dict['target catch point cm']
                                       / notes['cm_per_pixel'])
                    len_px = 5.0 / notes['cm_per_pixel']

                    tx_px = (center_x_px - throw_offset_px
                             if arc.hand_throw == 'right'
                             else center_x_px + throw_offset_px)
                    cx_px = (center_x_px - catch_offset_px
                             if arc.hand_catch == 'right'
                             else center_x_px + catch_offset_px)

                    tbx_dv, tby_dv = mapToDisplayVideo(
                            tx_px, center_y_px + len_px)
                    ttx_dv, tty_dv = mapToDisplayVideo(
                            tx_px, center_y_px - len_px)
                    tbx, tby = self.mapToView(tbx_dv, tby_dv)
                    ttx, tty = self.mapToView(ttx_dv, tty_dv)

                    cbx_dv, cby_dv = mapToDisplayVideo(
                            cx_px, center_y_px + len_px)
                    ctx_dv, cty_dv = mapToDisplayVideo(
                            cx_px, center_y_px - len_px)
                    cbx, cby = self.mapToView(cbx_dv, cby_dv)
                    ctx, cty = self.mapToView(ctx_dv, cty_dv)

                    painter.setPen(QPen(Qt.red, 2))
                    painter.setBrush(Qt.red)
                    painter.setOpacity(1.0)
                    painter.drawLine(round(tbx), round(tby),
                                     round(ttx), round(tty))
                    painter.drawLine(round(cbx), round(cby),
                                     round(ctx), round(cty))

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

    def draw_framenum(self, viewport, painter, framenum):
        """
        Called from paintEvent().

        Draw the frame number in lower left corner of the viewport.
        """
        vc = self.window.currentVideoItem.vc

        digits = 1 if framenum <= 1 else 1 + floor(log10(framenum))
        digits_max = 1 if vc.frames <= 1 else 1 + floor(log10(vc.frames - 1))

        movie_lower_left_x, movie_lower_left_y = self.mapToView(
                0.0, vc.graphicsvideoitem.size().height())
        lower_left_x = max(0, movie_lower_left_x)
        lower_left_y = min(viewport.size().height(), movie_lower_left_y)

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

    def mapToView(self, x, y):
        """
        Map from video coordinates (pixels) to view coordinates
        """
        vc = self.window.currentVideoItem.vc

        movie_coord = QPointF(x, y)
        scene_coord = vc.graphicsvideoitem.mapToScene(movie_coord)
        view_coord = self.mapFromScene(scene_coord)
        return view_coord.x(), view_coord.y()

    # -------------------------------------------------------------------------
    #  QGraphicsView overrides
    # -------------------------------------------------------------------------

    def paintEvent(self, e):
        """
        Draw the video frame with overlay graphics on top. Also do certain
        aspects of playback control that we want to be frame-accurate.
        """
        super().paintEvent(e)       # draw frame of video

        if self.window is None or self.window.currentVideoItem is None:
            return

        moviestatus = self.window.currentVideoItem.vc.player.mediaStatus()
        if (moviestatus != QMediaPlayer.BufferedMedia and
                moviestatus != QMediaPlayer.BufferingMedia and
                moviestatus != QMediaPlayer.EndOfMedia):
            return

        framenum = self.window.framenum()
        viewport = self.viewport()
        painter = QPainter(viewport)

        self.do_playback_control(framenum)
        self.draw_overlays(painter, framenum)
        self.draw_framenum(viewport, painter, framenum)

    def resizeEvent(self, e):
        if self.window is not None:
            if self.videosnappedtoframe:
                scene = self.scene()
                if scene is not None:
                    self.fitInView(scene.itemsBoundingRect(),
                                   Qt.KeepAspectRatio)
            elif self.window.currentVideoItem is not None:
                # see if the entire movie frame is now visible; if so then
                # snap to frame
                portRect = self.viewport().rect()
                scenePolygon = self.mapToScene(portRect)
                graphicsvideoitem = (self.window.currentVideoItem
                                     .vc.graphicsvideoitem)
                itemPolygon = graphicsvideoitem.mapFromScene(scenePolygon)
                itemRect = itemPolygon.boundingRect()
                if itemRect.contains(graphicsvideoitem.boundingRect()):
                    self.fitInView(graphicsvideoitem.boundingRect(),
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
    Subclass of QListWidget that implements drag-and-drop of files onto the
    list.

    We also attach a HEVideoContext object to each item in this list, into
    which all per-video data is stored in Hawkeye. See addVideo() below.
    """
    def __init__(self, main_window):
        super().__init__(parent=main_window)
        self.window = main_window
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def addVideo(self, filepath):
        if not os.path.isfile(filepath):
            return None

        item = QListWidgetItem(os.path.basename(filepath))
        item.foreground = item.foreground()   # save so we can restore later
        item.setForeground(Qt.gray)
        item.setFlags(item.flags() | Qt.ItemIsSelectable)

        item.vc = HEVideoContext(self.window, filepath)

        self.addItem(item)

        # signal worker to process the video
        self.window.sig_process_video.emit(filepath)
        self.window.worker_processing_queue_length += 1
        self.window.setWorkerBusyIcon()
        return item

    # -------------------------------------------------------------------------
    #  QListWidget overrides
    # -------------------------------------------------------------------------

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [os.path.abspath(url.toLocalFile())
                 for url in e.mimeData().urls()]
        paths.sort(key=lambda p: os.path.basename(p))

        for path in paths:
            workerfree = not self.window.isWorkerBusy()
            self.addVideo(path).setSelected(workerfree)

    def sizeHint(self):
        return QSize(150, 100)

# -----------------------------------------------------------------------------


class HEVideoContext(QObject):
    """
    Class to hold all the data for an individual video. It also has its own
    instance of QMediaPlayer.

    We first tried a single app-wide QMediaPlayer instance where we swapped out
    the media on each switch between videos, but it was never reliable.
    """
    def __init__(self, main_window, filepath):
        super().__init__(parent=main_window)
        self.window = main_window

        self.player = QMediaPlayer(parent=main_window,
                                   flags=QMediaPlayer.VideoSurface)
        self.player.stateChanged.connect(self.mediaStateChanged)
        self.player.error.connect(self.handlePlayerError)

        self.graphicsvideoitem = QGraphicsVideoItem()
        self.graphicsscene = QGraphicsScene(parent=self.window.view)
        self.graphicsscene.addItem(self.graphicsvideoitem)
        self.graphicsvideoitem.nativeSizeChanged.connect(
                self.videoNativeSizeChanged)

        self.player.setVideoOutput(self.graphicsvideoitem)

        self.filepath = filepath        # path to original video file
        self.notes = None               # notes dictionary
        self.videopath = None           # path to transcoded display video
        self.videoresolution = 0        # resolution of display video (vert.)
        self.output = ''                # output from scanner
        self.doneprocessing = False
        self.position = 0               # in milliseconds
        self.duration = None            # in milliseconds
        self.frames = None              # frames number from 0 to frames-1
        self.processing_step = 0        # for progress bar during scanning
        self.processing_steps_total = 0
        self.has_played = False         # see note in HEMainWindow.playMovie()
        self.map = None         # see HEMainWindow.on_worker_displayvid_done()
        self.overlays = True            # whether to draw video overlays

    def mediaStateChanged(self, state):
        """
        Signaled by the QMediaPlayer when the state of the media changes from
        paused to playing, and vice versa.
        """
        if self.isActive():
            if self.player.state() == QMediaPlayer.PlayingState:
                self.window.playButton.setIcon(
                        self.window.style().standardIcon(QStyle.SP_MediaPause))
            else:
                self.window.playButton.setIcon(
                        self.window.style().standardIcon(QStyle.SP_MediaPlay))

    @Slot()
    def handlePlayerError(self):
        """
        Called by the media player when there is an error.
        """
        if self.isActive():
            self.window.playButton.setEnabled(False)
            err = self.player.errorString()
            code = self.player.error()
            self.window.playerErrorLabel.setText(f'Error: {err} (code {code})')

    @Slot(int)
    def videoNativeSizeChanged(self, size):
        """
        Signaled when the video size changes.
        """
        self.graphicsvideoitem.setSize(size)

        if self.isActive():
            self.window.view.fitInView(self.graphicsvideoitem.boundingRect(),
                                       Qt.KeepAspectRatio)
            self.window.zoomInButton.setEnabled(True)
            self.window.zoomOutButton.setEnabled(False)
            self.window.view.videosnappedtoframe = True

    def isActive(self):
        """
        Returns True if this video is currently active in the main window,
        False otherwise.
        """
        return (self.window.currentVideoItem is not None
                and self.window.currentVideoItem.vc is self)


# -----------------------------------------------------------------------------

class HEViewList(QListWidget):
    """
    Subclass of standard QListWidget to show run-related info for a video. I
    wonder why QWidget doesn't have a setSizeHint() method?
    """
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def sizeHint(self):
        return QSize(150, 100)

# -----------------------------------------------------------------------------


class HETableViewDelegate(QStyledItemDelegate):
    """
    Subclass of QStyledItemDelegate to draw horizontal lines between separate
    runs in our Data View QTableWidget.
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
