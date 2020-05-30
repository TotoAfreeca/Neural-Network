from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from neural_network import NeuralNetwork
import numpy as np

import sys

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()

        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle("Neural Network")
        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText("LABEL1")
        self.label.move(50, 50)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Click me")
        self.b1.clicked.connect(self.clicked)

    def clicked(self):
        x = np.array([1, 2, 3])
        x = NeuralNetwork.sigmoid_unipolar_function()
        self.label.setText(np.array_str(x))
        self.update()

    def update(self):
        self.label.adjustSize()

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())


window()
