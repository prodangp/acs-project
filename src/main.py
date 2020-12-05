from PyQt5 import QtWidgets
from mainwindow import Ui_MainWindow

class GUI(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(self.__class__, self).__init__()
        screen = app.primaryScreen()
        size = screen.size()
        self.resize(int(size.width() * 0.75), int(size.height() * 0.75))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    guiapp = GUI()
    screen = app.primaryScreen()


    if guiapp.exec_() == QtWidgets.QDialog.Accepted:
        window = GUI()
        window.show()
        sys.exit(app.exec_())
