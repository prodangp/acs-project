# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'frgui.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import qtawesome as qta
import cv2
from test import test_image as test
from plots import plot
import hapt_table
class Ui_MainWindow(object):

    def upload_photo(self, label):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Upload photo", "",
                                                  "All Files (*)", options=options)

        if filename:
            self.image_path = filename
            image = cv2.imread(filename)
            image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                                      QtGui.QImage.Format_RGB888).rgbSwapped()
            label.setPixmap(QtGui.QPixmap.fromImage(
                image.scaled(int(label.width()), int(label.height()))))
            self.upload_button.setVisible(False)

    def show_stats(self):
        plot(self.comboBox.currentText())

    def test_image(self):
        training_images = self.horizontalSlider.value()
        self.horizontalSlider.setEnabled(False)
        algorithm = self.comboBox.currentText()
        norm = self.comboBox_2.currentText()
        if self.image_path:
            filename = test(self.image_path, algorithm, norm, training_images)
            image = cv2.imread(filename)
            image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                                      QtGui.QImage.Format_RGB888).rgbSwapped()
            self.result_image_label.setPixmap(QtGui.QPixmap.fromImage(
                image.scaled(int(self.result_image_label.width()), int(self.result_image_label.height()))))
        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('You have to upload an image!')
            error_dialog.exec_()

    def open_kmeans(self):
        kmeans = hapt_table.Ui_MainWindow()
        kmeans.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        width = 1024
        height = 512
        MainWindow.resize(width, height)
        self.image_path = None
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(226, 240, 232))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(226, 240, 232))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(226, 240, 232))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        self.centralwidget.setPalette(palette)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setHorizontalSpacing(8)
        self.gridLayout.setObjectName("gridLayout")
        self.result_image_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.result_image_label.sizePolicy().hasHeightForWidth())
        self.result_image_label.setSizePolicy(sizePolicy)
        self.result_image_label.setMinimumSize(QtCore.QSize(200, 200))
        self.result_image_label.setObjectName("result_image_label")
        self.gridLayout.addWidget(self.result_image_label, 0, 2, 1, 1)
        self.user_image_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.user_image_label.sizePolicy().hasHeightForWidth())
        self.user_image_label.setSizePolicy(sizePolicy)
        self.user_image_label.setMinimumSize(QtCore.QSize(200, 200))
        self.user_image_label.setText("")
        self.user_image_label.setObjectName("user_image_label")
        self.gridLayout.addWidget(self.user_image_label, 0, 1, 1, 1)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setAlignment(QtCore.Qt.AlignRight)
        self.fr_label = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fr_label.sizePolicy().hasHeightForWidth())
        self.fr_label.setSizePolicy(sizePolicy)
        self.fr_label.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("Sitka Text")
        font.setPointSize(12)
        self.fr_label.setFont(font)
        self.fr_label.setObjectName("fr_label")
        self.verticalLayout.addWidget(self.fr_label)
        self.comboBox = QtWidgets.QComboBox(self.widget)
        self.comboBox.setMaximumSize(QtCore.QSize(200, 36))
        font = QtGui.QFont()
        font.setFamily("Sitka")
        font.setPointSize(10)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout.addWidget(self.comboBox)
        self.comboBox_2 = QtWidgets.QComboBox(self.widget)
        self.comboBox_2.setMaximumSize(QtCore.QSize(200, 36))
        font = QtGui.QFont()
        font.setFamily("Sitka")
        font.setPointSize(10)
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.verticalLayout.addWidget(self.comboBox_2)
        self.stats_button = QtWidgets.QPushButton('Statistics', self.widget)
        self.verticalLayout.addWidget(self.stats_button)
        self.stats_button.setMaximumSize(QtCore.QSize(115, 40))
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.training_label = QtWidgets.QLabel(self.widget)
        self.training_label.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("Sitka Text")
        font.setPointSize(12)
        self.training_label.setFont(font)
        self.training_label.setObjectName("training_label")
        self.verticalLayout.addWidget(self.training_label)
        self.horizontalSlider = QtWidgets.QSlider(self.widget)
        self.horizontalSlider.setPalette(palette)
        self.horizontalSlider.setAcceptDrops(False)
        self.horizontalSlider.setMinimum(1)
        self.horizontalSlider.setMaximum(10)
        self.horizontalSlider.setPageStep(1)
        self.horizontalSlider.setProperty("value", 8)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setInvertedAppearance(False)
        self.horizontalSlider.setInvertedControls(False)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider.setTickInterval(1)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.verticalLayout.addWidget(self.horizontalSlider)
        self.test_button = QtWidgets.QPushButton('Test', self.widget)
        self.verticalLayout.addWidget(self.test_button)
        self.test_button.setMaximumSize(QtCore.QSize(80, 40))
        self.gridLayout.addWidget(self.widget, 0, 3, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1031, 26))
        self.menubar.setObjectName("menubar")
        self.menuk_Means = QtWidgets.QMenu(self.menubar)
        self.menuk_Means.setObjectName("menuk_Means")
        self.menuk_Means.mouseDoubleClickEvent = lambda event: self.open_kmeans()
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuk_Means.menuAction())

        self.upload_button = QtWidgets.QPushButton('', MainWindow)
        self.upload_button.move(int(width/5), int(height/2))
        upload_icon = qta.icon('fa.upload')
        self.upload_button.setIcon(upload_icon)
        self.upload_button.setIconSize(QtCore.QSize(36, 36))
        self.upload_button.setProperty("class", "iconButton")
        self.upload_button.clicked.connect(lambda: self.upload_photo(self.user_image_label))
        self.user_image_label.mousePressEvent = lambda event: self.upload_photo(self.user_image_label)

        graph_icon = qta.icon('ei.graph')
        self.stats_button.setIcon(graph_icon)
        self.stats_button.setIconSize(QtCore.QSize(50, 50))
        self.stats_button.setProperty("class", "iconButton")
        self.stats_button.clicked.connect(lambda: self.show_stats())

        test_icon = qta.icon('fa5.user')
        self.test_button.setIcon(test_icon)
        self.test_button.setIconSize(QtCore.QSize(40, 40))
        self.test_button.setProperty("class", "iconButton")
        self.test_button.clicked.connect(lambda: self.test_image())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face recognition GUI"))
        self.result_image_label.setText(_translate("MainWindow", ""))
        self.fr_label.setText(_translate("MainWindow", "Algorithm"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Nearest neighbours"))
        self.comboBox.setItemText(1, _translate("MainWindow", "k-Nearest neighbours"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Eigenfaces"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Eigenfaces 2"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Lanczos"))
        self.comboBox.setItemText(5, _translate("MainWindow", "Tensori A1"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Euclidean"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Manhattan"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "Infinity"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "Cosine"))
        self.training_label.setText(_translate("MainWindow", "Training/Test"))
        self.menuk_Means.setTitle(_translate("MainWindow", "k-Means"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())