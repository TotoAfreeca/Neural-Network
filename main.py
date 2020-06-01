import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLineEdit, QFormLayout, QWidget, \
    QTabWidget, QHBoxLayout, QSpinBox, QLabel, QDoubleSpinBox, QTextEdit, QRadioButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtCore, QtGui
# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from neural_network import NeuralNetwork
import numpy as np

class Window(QTabWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # Windows
        self.create_tab = QWidget()
        self.addTab(self.create_tab, "Create")

        # Create Window
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00, 1.00)
        self.learning_rate.setValue(0.1)
        self.learning_rate.setSingleStep(0.01)
        self.hidden_layers_number = QSpinBox()
        self.hidden_layers_number.setRange(0, 25)
        self.hidden_layers_number.setValue(2)
        self.hidden_layers_number.setSingleStep(1)
        regex = r"^(\s*(-|\+)?\d+(?:\.\d+)?\s*,\s*)+(-|\+)?\d+(?:\.\d+)?\s*$"
        validator = QtGui.QRegExpValidator(QtCore.QRegExp(regex), self)
        self.layers_line_edit = QLineEdit()
        self.layers_line_edit.setValidator(validator)




        self.function = QLineEdit()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.create_tab_ui()

    def create_tab_ui(self):

        button = QPushButton('Show function')
        toolbar = NavigationToolbar(self.canvas, self)
        data_form = QFormLayout()
        function_data = QHBoxLayout()
        function_data.addWidget(self.learning_rate)
        function_data.addWidget(QLabel('Coma separated layers sizes:'))
        function_data.addWidget(self.layers_line_edit)
        data_form.addRow('Learning rate:', function_data)
        data_form.addRow(toolbar)
        data_form.addRow(button)

        self.create_tab.setLayout(data_form)

    def select_option(self, b):

        if b.text() == "Max":
            if b.isChecked() == True:
                self.find_max = 1
            else:
                self.find_max = 0

        if b.text() == "Min":
            if b.isChecked() == True:
                self.find_max = 0
            else:
                self.find_max = 1




    def on_clicked(self):
        self._list_widget.clear()
        if self._le.text():
            values = [int(val) for val in self._le.text().split(",")]
            print(values)
            self.layer_sizes.addItems([str(val) for val in values])

def window():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())


window()
