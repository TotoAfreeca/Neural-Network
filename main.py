import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLineEdit, QFormLayout, QWidget, \
    QTabWidget, QHBoxLayout, QSpinBox, QLabel, QDoubleSpinBox, QTextEdit, QRadioButton, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtCore, QtGui
# Plotting
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

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
        regex = r"^(\s*(\+)?\d+(?:\.\d)?\s*,\s*)+(-|\+)?\d+(?:\.\d+)?\s*$"
        #regex = r"/^(\s*|\d+)$/"
        validator = QtGui.QRegExpValidator(QtCore.QRegExp(regex), self)
        self.layers_line_edit = QLineEdit()
        self.layers_line_edit.textEdited.connect(self.layers_number_change)
        self.layers_line_edit.setValidator(validator)

        self.file_button = QPushButton("Read file", self)
        self.file_button.clicked.connect(self.open_file_dialog)
        self.file_text_edit = QLineEdit()
        self.file_text_edit.setReadOnly(True)
        self.figure = plt.figure(figsize=(100, 100))
        self.canvas = FigureCanvas(self.figure)


        self.create_tab_ui()

    def create_tab_ui(self):


        toolbar = NavigationToolbar(self.canvas, self)
        data_form = QFormLayout()
        function_data = QHBoxLayout()
        function_data.addWidget(self.learning_rate)
        function_data.addWidget(QLabel('Coma separated layers sizes:'))
        function_data.addWidget(self.layers_line_edit)
        data_form.addRow('Learning rate:', function_data)

        file_data = QHBoxLayout()
        file_data.addWidget(self.file_button)
        file_data.addWidget(self.file_text_edit)
        data_form.addRow(file_data)
        data_form.addRow(toolbar)
        network_plot = QHBoxLayout()
        network_plot.addWidget(QLabel(""))
        network_plot.addWidget(self.canvas)
        network_plot.addWidget(QLabel(""))
        data_form.addRow(network_plot)

        button = QPushButton('CREATE')
        button.clicked.connect(self.create_network)
        data_form.addRow(button)
        self.layer_sizes = []
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


    def open_file_dialog(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file')

        try:
            if filename[0]:
                self.data = pd.read_csv(filename[0])
                self.file_text_edit.setText(filename[0])
        except ValueError:
            return
            #WRONG FILE
        return

    def layers_number_change(self):
        if self.layers_line_edit.text():
            self.layer_sizes = [int(val) for val in self.layers_line_edit.text().split(",") if val and val not in '0+ ']
            print(self.layer_sizes)






    def create_network(self):

        input_size = 3
        output_size = 4

        self.network = NeuralNetwork()
        self.network.create_layers(input_size, output_size, self.layer_sizes, self.network.sigmoid_unipolar_function,
                                   self.network.sigmoid_unipolar_prime)

        self.figure.clear()
        G = nx.DiGraph()

        layer_size = 0
        # input layer to first layer edges
        ed = []
        for i in range(0, self.network.layers[0].input_size):
            vertex_name = 'x'+str(i+1)
            for j in range(0, self.network.layers[0].output_size):
                ed.append([vertex_name, 'h'+str(1)+str(j+1), np.round(self.network.layers[0].weights[i, j], 3)])

        for layer_number in range(1, len(self.network.layers)):
            prev_layer_size = self.network.layers[layer_number-1].output_size
            for i in range(0, prev_layer_size):
                vertex_name = 'h'+str(layer_number)+str(i+1)
                for j in range(0, self.network.layers[layer_number].output_size):
                    if layer_number == len(self.network.layers)-1:
                        ed.append([vertex_name, 'OUT'+str(j+1), np.round(self.network.layers[layer_number].weights[i, j], 3)])
                    else:
                        ed.append([vertex_name, 'h' + str(layer_number + 1) + str(j + 1),
                                   np.round(self.network.layers[layer_number].weights[i, j], 3)])

        # ed = insert(ed, 2,list(np.around(self.network.layers[0].weights, 3).flatten()) , axis=1)
        print(ed)


        # ed = [['x1', 4, -1],
        #       ['x1', 5, -1],
        #       ['x2', 4, -1],
        #       ['x2', 5, -1],
        #       ['x3', 4, -1],
        #       ['x3', 5, 10],
        #       [4, 3, -1],
        #       [5, 3, 100]]

        G.add_weighted_edges_from(ed)
        pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
        nx.draw(G, with_labels=True, pos=pos, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'weight')

        nx.draw_networkx_edge_labels(G, pos=pos, font_weight='bold', label_pos=0.8, edge_labels=edge_labels)


        self.canvas.draw()



def window():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())


window()
