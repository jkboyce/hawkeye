# HEWidgets.py
#
# Custom Qt widgets for the Hawkeye application.
#
# Copyright 2019 Jack Boyce (jboyce@gmail.com)

import os
import platform
from math import log10, floor

from PySide2.QtCore import QSize, Qt, QPoint, QPointF
from PySide2.QtGui import QPainter, QPainterPath, QPen
from PySide2.QtWidgets import (QWidget, QGraphicsView, QGraphicsScene,
                               QListWidget, QListWidgetItem, QAbstractItemView,
                               QStyledItemDelegate)
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
        self.videosnappedtoframe = False

    def paintEvent(self, e):
        # draw the video frame
        super().paintEvent(e)

        if self.window is None:
            return
        videoitem = self.window.currentVideoItem
        if videoitem is None:
            return
        notes = videoitem._notes
        prefs = self.window.prefs

        # record current position in movie, so we can cue back if we switch
        # between movies
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
        frames_total = videoitem._frames

        # the following is a hack to solve the way QMediaPlayer on Windows
        # reports positions within videos, so we don't get an off-by-one error
        if platform.system() == 'Windows':
            notes_framenum = framenum
            # notes_framenum = framenum + 1
            # seems to be working correctly now...
        else:
            notes_framenum = framenum

        # check if we're at the beginning or end of the movie
        if framenum == 0:
            self.window.backButton.setEnabled(False)
        elif (framenum >= (frames_total - 1) or
              moviestatus == QMediaPlayer.EndOfMedia):
            # print('hit end of movie')
            self.window.mediaPlayer.play()
            self.window.mediaPlayer.pause()
            self.window.setFramenum(frames_total - 1)
            framenum = frames_total - 1
            self.window.backButton.setEnabled(videoitem._has_played)
            self.window.forwardButton.setEnabled(False)
        else:
            can_step = (self.window.mediaPlayer.state() !=
                        QMediaPlayer.PlayingState and videoitem._has_played)
            self.window.backButton.setEnabled(can_step)
            self.window.forwardButton.setEnabled(can_step)

        # do bounds clipping on stepping limits, and also check if we've hit
        # the end of a range we're stepping through
        if self.window.stepForwardUntil is not None:
            if self.window.stepForwardUntil >= frames_total:
                self.window.stepForwardUntil = frames_total - 1
            if framenum >= self.window.stepForwardUntil:
                if framenum > self.window.stepForwardUntil:
                    self.window.setFramenum(self.window.stepForwardUntil)
                self.window.stepForwardUntil = None
        if self.window.stepBackwardUntil is not None:
            if self.window.stepBackwardUntil < 0:
                self.window.stepBackwardUntil = 0
            if framenum <= self.window.stepBackwardUntil:
                if framenum < self.window.stepBackwardUntil:
                    self.window.setFramenum(self.window.stepBackwardUntil)
                self.window.stepBackwardUntil = None

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

        if notes['step'] >= 5:
            # things to draw when the full scan has been completed
            if videoitem._videoresolution == 0:
                def mapToDisplayVideo(x, y):
                    return x, y
            else:
                orig_height = notes['frame_height']
                orig_width = notes['frame_width']
                display_height = videoitem._videoresolution
                display_width = round(orig_width * display_height
                                      / orig_height)
                if display_width % 2 == 1:
                    display_width += 1

                def mapToDisplayVideo(x, y):
                    new_x = x * display_width / orig_width
                    new_y = y * display_height / orig_height
                    return new_x, new_y

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
            if (self.window.prefs['markers'] and notes_framenum
                    in notes['meas']):
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
                    painter.drawEllipse(QPoint(center_x, center_y),
                                        radius, radius)

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
                            painter.fillRect(arc_x+15, arc_y-5, 25, 9,
                                             Qt.black)
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
        digits_max = (1 if frames_total <= 1
                      else 1 + floor(log10(frames_total - 1)))
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

        if self.window.stepForwardUntil is not None:
            self.window.stepForward()
        elif self.window.stepBackwardUntil is not None:
            self.window.stepBackward()

    def mapToView(self, x, y):
        """
        map coordinate from movie coordinates (pixels) to view coordinates
        """
        movie_coord = QPointF(x, y)
        scene_coord = self.window.item.mapToScene(movie_coord)
        view_coord = self.mapFromScene(scene_coord)
        return view_coord.x(), view_coord.y()

    def resizeEvent(self, e):
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
    Subclass of QListWidget that implements drag-and-drop of files onto the
    list.
    """
    def __init__(self, main_window):
        super().__init__(parent=main_window)
        self.window = main_window
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
        item._videopath = None
        item._videoresolution = 0
        item._foreground = item.foreground()
        item._output = ''
        item._doneprocessing = False
        item._position = 0              # in milliseconds
        item._duration = None           # in milliseconds
        item._frames = None             # frames number from 0 to frames-1
        item._processing_step = 0
        item._processing_steps_total = 0
        item._has_played = False        # see note in HEMainWindow.playMovie()
        """
        QGraphicsVideoItem and QGraphicsScene have scaling problems when
        the video changes. So we keep a separate per-video instance of each
        and swap them in when we change videos in the view. See
        switchPlayerToVideoItem().
        """
        item._graphicsvideoitem = QGraphicsVideoItem()
        item._graphicsscene = QGraphicsScene(self.window.view)
        item._graphicsscene.addItem(item._graphicsvideoitem)
        item._graphicsvideoitem.nativeSizeChanged.connect(
                self.window.videoNativeSizeChanged)
        item.setForeground(Qt.gray)
        item.setFlags(item.flags() | Qt.ItemIsSelectable)
        self.addItem(item)

        self.window.sig_new_work.emit(filepath)
        self.window.worker_queue_length += 1
        return item

    def sizeHint(self):
        return QSize(150, 100)

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
