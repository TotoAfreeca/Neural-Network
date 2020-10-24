import pandas as pd
import numpy as np

from sklearn import preprocessing


# Simple class to perform basic data encoding, train/test splitting etc. on a file
# note that the data need to be learning ready - means it has to contain all the data of appropriate format
# aside from the target variable - it is automatically encoded into multiple column variable using one-hot-encoding

class DataFormatter:

    def __init__(self, csv_file_name):
        self.dataframe = pd.read_csv(csv_file_name)
        self.dataframe = self.dataframe.reindex(np.random.permutation(self.dataframe.index))
        self.input_size = 0
        self.output_size = 0

        mm_scaler = preprocessing.MinMaxScaler()


        train = self.dataframe.sample(frac=0.8)


        test = self.dataframe.drop(train.index)


        self.input_size = len(self.dataframe.columns) - 1
        self.x_train = train.iloc[:, :-1].values
        self.x_train = mm_scaler.fit_transform(self.x_train)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.input_size)
        self.x_train = self.x_train.astype('float32')


        self.y_train = train.iloc[:, -1].values
        onehot = pd.get_dummies(self.y_train)
        target_labels = onehot.columns
        self.y_train = onehot.values

        self.x_test = test.iloc[:, :-1].values
        self.x_test = mm_scaler.fit_transform(self.x_test)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.input_size)
        self.x_test = self.x_test.astype('float32')

        self.y_test = test.iloc[:, -1].values
        onehot = pd.get_dummies(self.y_test)
        target_labels = onehot.columns
        self.y_test = onehot.values

        self.output_size = len(onehot.columns)

    def get_training_set(self):
        return self.x_train, self.y_train

    def get_test_set(self):
        return self.x_test, self.y_test

    def get_sizes(self):
        return self.input_size, self.output_size

    def get_headers(self):
        return self.dataframe.columns[0:-1]

