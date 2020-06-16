import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLineEdit, QFormLayout, QWidget, \
    QTabWidget, QHBoxLayout, QSpinBox, QLabel, QDoubleSpinBox, QTextEdit, QRadioButton, QFileDialog, QErrorMessage, \
    QPlainTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtCore, QtGui
# Plotting
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from DataFormatter import DataFormatter
import time
from networkx.drawing.nx_agraph import graphviz_layout

from matplotlib import cm, colors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from NeuralNetwork import NeuralNetwork
import numpy as np

class Window(QTabWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        np.set_printoptions(suppress=True)
        np.set_printoptions(linewidth=np.inf)
        # Windows
        self.create_tab = QWidget()
        self.addTab(self.create_tab, "Create")

        self.train_tab = QWidget()
        self.addTab(self.train_tab, "Train")

        self.error_tab = QWidget()
        self.addTab(self.error_tab, "Error")

        self.summary_tab = QWidget()
        self.addTab(self.summary_tab, "Summary")

        self.error_list = []
        self.test_error = []

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
        self.network_exists = False
        self.input_size = 0
        self.output_size = 0

        self.unipolar = QRadioButton("Unipolar sigmoid function")
        self.unipolar.toggled.connect(lambda: self.select_option(self.unipolar))
        self.unipolar.setChecked(True)
        self.tanh = QRadioButton("Hyperbolic tangent function")
        self.tanh.toggled.connect(lambda: self.select_option(self.tanh))

        self.activation_function = self.sigmoid_unipolar_function
        self.activation_prime = self.sigmoid_unipolar_prime

        self.figure = plt.figure(num=1, figsize=(100, 100))
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
        self.max_error = QDoubleSpinBox()
        self.max_error.setRange(0.0001, 0.9999)
        self.max_error.setValue(0.09)

        self.epoch_sum = 0

        self.epoch_label = QLabel("Epoch: ")
        self.error_label = QLabel("Error: ")

        self.canvas_train = FigureCanvas(plt.figure(1))
        self.stop = False
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.change_stop)
        self.randomize_button = QPushButton('Initialize weights')
        self.randomize_button.clicked.connect(self.randomize)
        self.train_by_steps_button = QPushButton('Train ' + self.epochs_number.text() + ' epochs')


        #error tab


        self.epoch_label_error = QLabel("Epoch: ")
        self.error_label_error = QLabel("Error: ")


        self.error_figure = plt.figure(num=2, figsize=(100, 100))
        self.canvas_error = FigureCanvas(self.error_figure)



        self.stop = False
        self.stop_button_error = QPushButton("Stop")
        self.stop_button_error.clicked.connect(self.change_stop)
        self.randomize_button_error = QPushButton('Initialize weights')
        self.randomize_button_error.clicked.connect(self.randomize)
        self.train_by_steps_button_error = QPushButton('Train ' + self.epochs_number.text() + ' epochs')

        #summary tab
        self.summary = QPlainTextEdit()
        self.get_summary_button = QPushButton("Predict test & get summary")
        self.get_summary_button.clicked.connect(self.write_summary)

        self.train_tab_ui()
        self.create_tab_ui()
        self.error_tab_ui()
        self.summary_tab_ui()





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

        self.epochs_number.editingFinished.connect(self.epochs_number_edited)
        self.layer_sizes = []

        self.create_tab.setLayout(data_form)

    def train_tab_ui(self):
        toolbar_train = NavigationToolbar(self.canvas_train, self)
        train_form = QFormLayout()
        train_data = QHBoxLayout()
        train_data.addWidget(self.learning_rate)
        train_data.addWidget(QLabel('Expected error'))
        train_data.addWidget(self.max_error)
        train_data.addWidget(QLabel('No. epochs'))
        train_data.addWidget(self.epochs_number)
        train_form.addRow('Learning rate:', train_data)




        train_form.addRow(toolbar_train)

        epoch_row = QHBoxLayout()
        epoch_row.addWidget(self.epoch_label)
        epoch_row.addWidget(self.error_label)
        train_form.addRow(epoch_row)

        network_plot_train = QHBoxLayout()
        network_plot_train.addWidget(self.canvas_train)
        train_form.addRow(network_plot_train)

        buttons = QHBoxLayout()
        buttons.addWidget(self.randomize_button)
        buttons.addWidget(self.stop_button)
        self.train_by_steps_button.clicked.connect(self.gui_train)
        buttons.addWidget(self.train_by_steps_button)

        train_form.addRow(buttons)

        self.train_tab.setLayout(train_form)

    def error_tab_ui(self):

        error_form = QFormLayout()

        toolbar_error = NavigationToolbar(self.canvas_error, self)
        error_form.addRow(toolbar_error)

        epoch_row = QHBoxLayout()
        epoch_row.addWidget(self.epoch_label_error)
        epoch_row.addWidget(self.error_label_error)
        error_form.addRow(epoch_row)

        error_plot = QHBoxLayout()
        error_form.addWidget(self.canvas_error)
        error_form.addRow(error_plot)

        buttons = QHBoxLayout()
        buttons.addWidget(self.randomize_button_error)
        buttons.addWidget(self.stop_button_error)
        self.train_by_steps_button_error.clicked.connect(self.gui_train_error)
        buttons.addWidget(self.train_by_steps_button_error)

        error_form.addRow(buttons)

        self.error_tab.setLayout(error_form)

    def summary_tab_ui(self):
        summary_form = QFormLayout()
        summary_form.addWidget(self.summary)
        summary_form.addWidget(self.get_summary_button)

        self.summary_tab.setLayout(summary_form)


    def select_option(self, b):

        if b.text() == "Unipolar sigmoid function":
            if b.isChecked() == True:
                self.activation_function = self.sigmoid_unipolar_function
                self.activation_prime = self.sigmoid_unipolar_prime
            else:
                self.activation_function = self.tanh_function
                self.activation_prime = self.tanh_prime

        if b.text() == "Hyperbolic tangent function":
            if b.isChecked() == True:
                self.activation_function = self.tanh_function
                self.activation_prime = self.tanh_prime
            else:
                self.activation_function = self.sigmoid_unipolar_function
                self.activation_prime = self.sigmoid_unipolar_prime


    def open_file_dialog(self):
        dialog = QFileDialog.getOpenFileName(self, 'Open file')

        if dialog[0].endswith('.csv'):
                formatter = DataFormatter(dialog[0])
                self.x_train, self.y_train = formatter.get_training_set()
                self.x_test, self.y_test = formatter.get_test_set()
                self.input_size, self.output_size = formatter.get_sizes()
                self.file_text_edit.setText(dialog[0])
        else:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Please select csv file.')
            error_dialog.exec_()

    def layers_number_change(self):
        if self.layers_line_edit.text():
            self.layer_sizes = [int(val) for val in self.layers_line_edit.text().split(",") if val and val not in '0+ ']
            print(self.layer_sizes)




    def create_network(self):

        self.epoch_sum = 0
        self.error_list = []
        self.test_error = []
        self.network = NeuralNetwork()
        if self.input_size > 0 and self.output_size > 0:
            self.network.create_layers(self.input_size, self.output_size, self.layer_sizes, self.activation_function,
                                   self.activation_prime)

            self.timer = QtCore.QTimer()
            self.timer.setInterval(100)
            self.timer.timeout.connect(self.plot_error)
            self.plot_network(self.canvas_create)
            self.plot_network(self.canvas_train)
        else:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Please load the file first.')
            error_dialog.exec_()

    def plot_error(self):
        if not self.stop:
            plt.figure(2)
            self.error_figure.clear()
            test1 = np.arange(self.epoch_sum)
            test2 = self.error_list
            test3 = self.test_error
            plt.plot(test1, test2, label='train')
            #plt.plot(test1, test3, label='test')
            plt.xlabel("Epoch")
            plt.ylabel("Mean squared error (MSE)")
            plt.legend(loc='upper right')
            plt.grid()

            self.update_labels()
            self.canvas_error.draw()

    def plot_network(self, canvas):
        plt.figure(1)
        self.figure.clear()
        G = nx.DiGraph()

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

        self.update_labels()
        self.canvas_train.draw()
        self.figure
        self.canvas_create.draw()




    def gui_train(self):

        self.stop = False
        self.timer.start()
        QApplication.processEvents()
        for i in range(int(self.epochs_number.value())):
            if(self.stop != True):
                self.network.train(self.x_train,
                               self.y_train,
                               epochs=1,
                               learning_rate=float(self.learning_rate.value()))

                self.epoch_sum += 1
                self.error_list.append(np.round(self.network.err, 5))
                #self.test_error.append(np.round(self.network.calculate_test_mse(self.x_test, self.y_test), 5))
                self.update_labels()
                self.plot_network(self.canvas_create)
                self.plot_network(self.canvas_train)
                QApplication.processEvents()
                if self.network.err <= self.max_error.value():
                    break
            else:
                self.plot_network(self.canvas_create)
                self.plot_network(self.canvas_train)
                break
        self.stop = True
        self.update_labels()
        self.plot_network(self.canvas_create)
        self.plot_network(self.canvas_train)
        self.timer.stop()

    def gui_train_error(self):
        self.stop = False


        self.timer.start()

        QApplication.processEvents()
        for i in range(int(self.epochs_number.value())):
            if (self.stop != True):
                self.network.train(self.x_train,
                                   self.y_train,
                                   epochs=1,
                                   learning_rate=float(self.learning_rate.value()))

                self.epoch_sum += 1
                self.error_list.append(np.round(self.network.err, 5))
                #self.test_error.append(np.round(self.network.calculate_test_mse(self.x_test, self.y_test), 5))
                if self.network.err <= self.max_error.value():
                    break
                #self.plot_error()
                QApplication.processEvents()
        self.stop = True
        self.plot_network(self.canvas_create)
        self.plot_network(self.canvas_train)
        self.timer.stop()


    def update_labels(self):
        self.epoch_label.setText("Epoch: " + str(self.epoch_sum))
        self.error_label.setText("Error: " + str(np.round(self.network.err, 5)))
        self.epoch_label_error.setText("Epoch: " + str(self.epoch_sum))
        self.error_label_error.setText("Error: " + str(np.round(self.network.err, 5)))

    def epochs_number_edited(self):
        self.train_by_steps_button.setText('Train ' + self.epochs_number.text() + ' epochs')
        self.train_by_steps_button_error.setText('Train ' + self.epochs_number.text() + ' epochs')

    def randomize(self):
        self.stop = True
        self.epoch_sum = 0
        self.error_list = []
        self.test_error = []
        self.network.randomize_layers()
        self.plot_network(self.canvas_create)
        self.plot_network(self.canvas_train)
        self.canvas_error.draw()
        self.timer.stop()

    def write_summary(self):
        self.summary.clear()
        self.summary.appendPlainText("Finished learning")
        self.summary.appendPlainText("Epochs: " + str(self.epoch_sum))
        self.summary.appendPlainText("Train error: " + str(self.network.err))
        self.summary.appendPlainText("Test error: " + str(self.network.calculate_test_mse(self.x_test, self.y_test)))
        self.summary.appendPlainText("TEST RESULTS BELOW")
        self.summary.appendPlainText("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for i in range(len(self.x_test)):
            self.summary.appendPlainText(str(np.round(self.network.predict(self.x_test[i]),3)))
            self.summary.appendPlainText(str(self.y_test[i]))


    def sigmoid_unipolar_function(self, x):
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)

    def sigmoid_unipolar_prime(self, z):
        return self.sigmoid_unipolar_function(z) * (1 - self.sigmoid_unipolar_function(z))

    def tanh_function(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2

    def change_stop(self):
        self.stop = True

def window():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())


window()
