# -*-coding:utf-8 -*-
# Author   : zzp
# Date     : 2020/4/28 0:53
# Email AD ：2410520561@qq.com
# SoftWare : PyCharm
# Project Name   : SmartDog_v2
# Python Version : 3.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from PyQt5.QtWidgets import QMainWindow, QFileDialog
from SmartDogui import Ui_Smartdog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2


class Window(QMainWindow, Ui_Smartdog):

    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.target_rect = []
        self.video_path = ''
        self.FPS = 40
        self.initui()
        self.setSlots()
        self.load_track_algo()
        self.init_track()
        self.slot_press_camera()

    def initui(self):
        self.width = self.label_show.width()
        self.height = self.label_show.height()
        self.openPushBottom(camera=True, video=True)
        self.open_keyboard_flag = False

    def setSlots(self):
        self.btn_select_target.clicked.connect(self.slot_press_select_roi)
        # self.btn_open_camera.clicked.connect(self.slot_press_camera)
        # self.btn_open_video.clicked.connect(self.slot_press_video)
        self.btn_track_over.clicked.connect(self.slot_press_over)
        self.btn_track_start.clicked.connect(self.slot_press_track)

    def load_track_algo(self):
        self.config_path = '../models/siamrpn_mobilev2_l234_dwxcorr/config.yaml'
        self.snapshot_path = '../models/siamrpn_mobilev2_l234_dwxcorr/model.pth'
        self.textBws_show_process.append('The initialization of program is complete.')
        self.textBws_show_process.append('The track model siamrpn_mobilev2 has been loaded.')

    def openPushBottom(self, select_target=False, camera=False, video=False, over=False, track=False):
        self.btn_select_target.setEnabled(True) if select_target is True else self.btn_select_target.setEnabled(False)
        # self.btn_open_camera.setEnabled(True) if camera is True else self.btn_open_camera.setEnabled(False)
        # self.btn_open_video.setEnabled(True) if video is True else self.btn_open_video.setEnabled(False)
        self.btn_track_over.setEnabled(True) if over is True else self.btn_track_over.setEnabled(False)
        self.btn_track_start.setEnabled(True) if track is True else self.btn_track_start.setEnabled(False)

    def slot_press_camera(self):
        self.textBws_show_process.append('Open the Camera...')
        self.label_show.clear_flag = False
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened:
            self.textBws_show_process.append('Open the Camera succeed.')
        self.open_camera_flag = True
        self.open_video_flag = False
        self.camera_timer = QTimer(self)
        self.textBws_show_process.append('the thread has build.')
        self.camera_timer.timeout.connect(self.slot_camera_show_image)
        self.textBws_show_process.append('the camera is running...')
        self.camera_timer.start(self.FPS)
        self.openPushBottom(select_target=True, over=True)

    def slot_press_video(self):
        video_path = QFileDialog.getOpenFileName(self,
                                                 'choose your video',
                                                 os.path.dirname(__file__),
                                                 '*.mp4 *.avi')
        if video_path[0] == '':
            return
        self.video_path = video_path[0]
        self.textBws_show_process.append('The video path is: ' + self.video_path)
        self.textBws_show_process.append('load the video file succeed!')
        self.label_show.clear_flag = False
        self.open_video_flag = True
        self.open_camera_flag = False
        self.capture_video = cv2.VideoCapture(self.video_path)
        self.video_timer = QTimer(self)
        self.textBws_show_process.append('the thread has build.')
        self.video_timer.timeout.connect(self.slot_video_show_image)
        self.textBws_show_process.append('play the video...')
        self.video_timer.start(self.FPS)
        self.openPushBottom(select_target=True, over=True)

    def slot_press_select_roi(self):
        self.open_keyboard_flag = True
        if self.open_video_flag is True:
            self.first_frame = self.frame
            self.video_timer.stop()
        elif self.open_camera_flag is True:
            self.first_frame = self.frame
            self.camera_timer.stop()
        else:
            pass
        self.textBws_show_process.append(
            'please choose your track target: press "s" to choose and press "q" to confire...')
        self.openPushBottom(over=True, track=True)

    def init_track(self):
        cfg.merge_from_file(self.config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        self.textBws_show_process.append('Model object creation...')
        self.checkpoint = torch.load(self.snapshot_path, map_location=lambda storage, loc: storage.cpu())
        self.model = ModelBuilder()
        self.model.load_state_dict(self.checkpoint)
        self.model.eval().to(device)
        self.textBws_show_process.append('The load of the tracking model is complete.')
        self.tracker = build_tracker(self.model)

    def slot_press_track(self):
        if self.open_keyboard_flag is False:
            self.target_rect = self.transformrect(self.label_show.rect)
            self.textBws_show_process.append('Target box format transcoding...')
            init_rect = tuple(self.target_rect)
            self.tracker.init(self.first_frame, init_rect)
            self.textBws_show_process.append('The target box is initialized.')
            self.clear_label()
            self.textBws_show_process.append('The tracking interface is cleared...')
            self.tracker_timer = QTimer(self)
            self.textBws_show_process.append('The trace thread has been created.')
            self.tracker_timer.timeout.connect(self.slot_track_process)
            self.tracker_timer.start(self.FPS)
            self.textBws_show_process.append('The target has been selected.')
            self.openPushBottom(over=True)

    def slot_press_over(self):
        self.clear_label()
        # self.openPushBottom(video=True, camera=True)
        if self.open_camera_flag is True:
            self.camera_timer.stop()
            self.camera.release()
            self.textBws_show_process.append('The model has released.')
            if self.open_track_flag is True:
                self.tracker_timer.stop()
                self.camera.release()
                self.open_track_flag = False
            self.open_camera_flag = False
        if self.open_video_flag is True:
            self.video_timer.stop()
            self.capture_video.release()
            self.textBws_show_process.append('The model has released.')
            if self.open_track_flag is True:
                self.tracker_timer.stop()
                self.capture_video.release()
                self.open_track_flag = False
            self.open_video_flag = False
        self.textBws_show_process.append('The trace thread has been destroy.')
        self.textBws_show_process.append('The tracking interface has been cleaned up.')
        self.textBws_show_process.append('The track has over.')
        self.slot_press_camera()

    def slot_camera_show_image(self):
        if self.open_camera_flag is True:
            if self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret is True:
                    self.frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.camera_timer.stop()
                self.camera.release()
                self.textBws_show_process.append('Track exceptions.')
        else:
            self.camera_timer.stop()
            self.camera.release()
            self.textBws_show_process.append('Track exceptions.')

    def slot_video_show_image(self):
        if self.open_video_flag is True:
            if self.capture_video.isOpened() is True:
                ret, frame = self.capture_video.read()
                if ret is True:
                    self.frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.video_timer.stop()
                self.capture_video.release()
                self.textBws_show_process.append('Track exceptions.')
        else:
            self.video_timer.stop()
            self.capture_video.release()
            self.textBws_show_process.append('Track exceptions.')

    def slot_track_process(self):
        self.open_track_flag = True
        if self.open_video_flag is True:
            if self.capture_video.isOpened() is True:
                ret, frame = self.capture_video.read()
                if ret is True:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    outputs = self.tracker.track(frame)
                    if 'polygon' in outputs:
                        polygon = np.array(outputs['polygon']).astype(np.int32)
                        cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                      True, (0, 255, 0), 3)
                        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                        mask = mask.astype(np.uint8)
                        mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                    else:
                        bbox = list(map(int, outputs['bbox']))
                        cv2.rectangle(frame, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 3)
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.tracker_timer.stop()
                self.camera.release()
                self.textBws_show_process.append('Track exceptions.')
        elif self.open_camera_flag is True:
            if self.camera.isOpened() is True:
                ret, frame = self.camera.read()
                if ret is True:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    outputs = self.tracker.track(frame)
                    if 'polygon' in outputs:
                        polygon = np.array(outputs['polygon']).astype(np.int32)
                        cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                      True, (0, 255, 0), 3)
                        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                        mask = mask.astype(np.uint8)
                        mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                    else:
                        bbox = list(map(int, outputs['bbox']))
                        cv2.rectangle(frame, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 3)
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.tracker_timer.stop()
                self.capture_video.release()
                self.textBws_show_process.append('Track exceptions.')

    def clear_label(self):
        self.label_show.clear_flag = True
        self.label_show.clear()

    def transformrect(self, roi_rect):
        t_rect = [roi_rect.x(), roi_rect.y(),
                  roi_rect.width(), roi_rect.height()]
        return t_rect

    def keyPressEvent(self, QKeyEvent):
        if self.open_keyboard_flag == True:
            if QKeyEvent.key() == Qt.Key_S:
                self.label_show.setCursor(Qt.CrossCursor)
                self.label_show.open_mouse_flag = True
                self.label_show.draw_roi_flag = True
            if QKeyEvent.key() == Qt.Key_Q:
                self.label_show.unsetCursor()
                self.label_show.draw_roi_flag = False
                self.label_show.open_mouse_flag = False
                self.open_keyboard_flag = False
