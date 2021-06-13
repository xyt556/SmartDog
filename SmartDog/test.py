import os.path

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton
import sys


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.resize(400, 400)
        self.btn = QPushButton(self)
        self.btn.clicked.connect(self.getfileanme)

    def getfileanme(self):
        a, b = QFileDialog.getOpenFileName(self, "File", os.path.dirname(__file__), "*.mp4 *.avi")
        if a == '':
            print("a is none")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
