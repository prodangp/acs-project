# -*- coding: utf-8 -*-
# MainWindow Design
# Here are created all the layouts and starting widgets
# Last major design update: 10 July 2020


from PyQt5 import QtCore, QtWidgets, QtGui


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        # Create the central widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # Create the grid
        self.grid = QtWidgets.QGridLayout()

        # Create the layout for device selection widgets
        self.selectDeviceVerticalLayout = QtWidgets.QVBoxLayout()
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
