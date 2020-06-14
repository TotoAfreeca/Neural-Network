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

        self.train_tab = QWidget()
        self.addTab(self.train_tab, "Train")


        # Create Tab
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

        self.unipolar = QRadioButton("Unipolar sigmoid function")
        self.unipolar.toggled.connect(lambda: self.select_option(self.unipolar))
        self.unipolar.setChecked(True)
        self.tanh = QRadioButton("Hyperbolic tangent function")
        self.tanh.toggled.connect(lambda: self.select_option(self.tanh))

        self.activation = NeuralNetwork.sigmoid_unipolar_function
        self.activation_prime = NeuralNetwork.sigmoid_unipolar_prime

        self.figure = plt.figure(figsize=(100, 100))
        self.canvas_create = FigureCanvas(self.figure)



        #train tab

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.9999)
        self.learning_rate.setValue(0.1)
        self.learning_rate.setSingleStep(0.01)
        self.learning_rate.setDecimals(4)

        self.epochs_number = QSpinBox()
        self.epochs_number.setRange(1, 100000)
        self.epochs_number.setValue(100)
        self.epochs_number.setSingleStep(1)
        self.canvas_train = FigureCanvas(self.figure)



        self.train_tab_ui()
        self.create_tab_ui()


    def create_tab_ui(self):


        toolbar_create = NavigationToolbar(self.canvas_create, self)
        data_form = QFormLayout()
        network_data = QHBoxLayout()
        network_data.addWidget(self.layers_line_edit)
        data_form.addRow('Coma separated layers sizes:', network_data)

        file_data = QHBoxLayout()
        file_data.addWidget(self.file_button)
        file_data.addWidget(self.file_text_edit)

        functions_data = QHBoxLayout()
        functions_data.addWidget(self.unipolar)
        functions_data.addWidget(self.tanh)
        data_form.addRow(functions_data)

        data_form.addRow(file_data)
        data_form.addRow(toolbar_create)

        network_plot_create = QHBoxLayout()
        network_plot_create.addWidget(QLabel(""))
        network_plot_create.addWidget(self.canvas_create)
        network_plot_create.addWidget(QLabel(""))
        data_form.addRow(network_plot_create)

        button = QPushButton('CREATE')
        button.clicked.connect(self.create_network)
        data_form.addRow(button)
        self.layer_sizes = []

        self.create_tab.setLayout(data_form)

    def train_tab_ui(self):
        toolbar_train = NavigationToolbar(self.canvas_train, self)
        train_form = QFormLayout()
        train_data = QHBoxLayout()
        train_data.addWidget(self.learning_rate)
        train_data.addWidget(QLabel("No. epochs"))
        train_data.addWidget(self.epochs_number)
        train_form.addRow("Learning rate:", train_data)


        network_plot_train = QHBoxLayout()
        network_plot_train.addWidget(self.canvas_train)
        train_form.addRow(toolbar_train)
        train_form.addRow(network_plot_train)

        self.train_tab.setLayout(train_form)


    def select_option(self, b):

        if b.text() == "Unipolar sigmoid function":
            if b.isChecked() == True:
                self.activation_function = NeuralNetwork.sigmoid_unipolar_function
                self.activation_prime = NeuralNetwork.sigmoid_unipolar_prime
            else:
                self.activation_function = NeuralNetwork.tanh
                self.activation_prime = NeuralNetwork.tanh_prime

        if b.text() == "Hyperbolic tangent function":
            if b.isChecked() == True:
                self.activation_function = NeuralNetwork.tanh
                self.activation_prime = NeuralNetwork.tanh_prime
            else:
                self.activation_function = NeuralNetwork.sigmoid_unipolar_function
                self.activation_prime = NeuralNetwork.sigmoid_unipolar_prime


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
        self.network.create_layers(input_size, output_size, self.layer_sizes, self.activation,
                                   self.activation_prime)

        self.plot_network()


    def plot_network(self):

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
        # print(ed)


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

        nx.draw_networkx_edge_labels(G, pos=pos, font_weight='bold', label_pos=0.85, edge_labels=edge_labels)


        self.canvas_create.draw()
        self.canvas_train.draw()



def window():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())


window()
