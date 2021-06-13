# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_SmartDog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from Label import Label


class Ui_Smartdog(object):
    def setupUi(self, Smartdog):
        Smartdog.setObjectName("Smartdog")
        Smartdog.resize(1000, 800)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Smartdog.sizePolicy().hasHeightForWidth())
        Smartdog.setSizePolicy(sizePolicy)
        Smartdog.setMinimumSize(QtCore.QSize(1000, 800))
        self.centralwidget = QtWidgets.QWidget(Smartdog)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.widget.setStyleSheet("")
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        # self.label_show = QtWidgets.QLabel(self.widget)
        self.label_show = Label(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_show.sizePolicy().hasHeightForWidth())
        self.label_show.setSizePolicy(sizePolicy)
        self.label_show.setMinimumSize(QtCore.QSize(1000, 600))
        self.label_show.setSizeIncrement(QtCore.QSize(900, 800))
        self.label_show.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_show.setText("")
        self.label_show.setObjectName("label_show")
        self.gridLayout_2.addWidget(self.label_show, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.widget)
        self.groupBox_2.setMinimumSize(QtCore.QSize(900, 0))
        self.groupBox_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.btn_open_camera = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_open_camera.setMinimumSize(QtCore.QSize(220, 80))
        self.btn_open_camera.setMaximumSize(QtCore.QSize(220, 80))
        self.btn_open_camera.setObjectName("btn_open_camera")
        self.gridLayout.addWidget(self.btn_open_camera, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(163, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
        self.btn_open_video = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_open_video.setMinimumSize(QtCore.QSize(220, 80))
        self.btn_open_video.setMaximumSize(QtCore.QSize(220, 80))
        self.btn_open_video.setObjectName("btn_open_video")
        self.gridLayout.addWidget(self.btn_open_video, 0, 3, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(164, 17, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 0, 4, 1, 1)
        self.btn_open_camera.raise_()
        self.btn_open_video.raise_()
        self.gridLayout_2.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.widget, 0, 0, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.btn_select_target = QtWidgets.QPushButton(self.groupBox_4)
        self.btn_select_target.setMinimumSize(QtCore.QSize(140, 60))
        self.btn_select_target.setMaximumSize(QtCore.QSize(140, 60))
        self.btn_select_target.setObjectName("btn_select_target")
        self.gridLayout_5.addWidget(self.btn_select_target, 0, 0, 1, 1)
        self.btn_track_start = QtWidgets.QPushButton(self.groupBox_4)
        self.btn_track_start.setMinimumSize(QtCore.QSize(140, 60))
        self.btn_track_start.setMaximumSize(QtCore.QSize(140, 60))
        self.btn_track_start.setObjectName("btn_track_start")
        self.gridLayout_5.addWidget(self.btn_track_start, 1, 0, 1, 1)
        self.btn_track_over = QtWidgets.QPushButton(self.groupBox_4)
        self.btn_track_over.setMinimumSize(QtCore.QSize(140, 60))
        self.btn_track_over.setMaximumSize(QtCore.QSize(140, 60))
        self.btn_track_over.setObjectName("btn_track_over")
        self.gridLayout_5.addWidget(self.btn_track_over, 2, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_4, 0, 1, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.textBws_show_process = QtWidgets.QTextBrowser(self.groupBox_3)
        self.textBws_show_process.setObjectName("textBws_show_process")
        self.gridLayout_3.addWidget(self.textBws_show_process, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_3, 1, 0, 1, 2)
        self.gridLayout_4.setColumnStretch(0, 2)
        Smartdog.setCentralWidget(self.centralwidget)

        self.retranslateUi(Smartdog)
        QtCore.QMetaObject.connectSlotsByName(Smartdog)

    def retranslateUi(self, Smartdog):
        _translate = QtCore.QCoreApplication.translate
        Smartdog.setWindowTitle(_translate("Smartdog", "SmartDog"))
        self.groupBox_2.setTitle(_translate("Smartdog", "跟踪方式"))
        self.btn_open_camera.setText(_translate("Smartdog", "摄像头跟踪"))
        self.btn_open_video.setText(_translate("Smartdog", "视频文件跟踪"))
        self.groupBox_4.setTitle(_translate("Smartdog", "跟踪选择"))
        self.btn_select_target.setText(_translate("Smartdog", "选择跟踪目标"))
        self.btn_track_start.setText(_translate("Smartdog", "开始跟踪"))
        self.btn_track_over.setText(_translate("Smartdog", "结束跟踪"))
        self.groupBox_3.setTitle(_translate("Smartdog", "进程显示"))
