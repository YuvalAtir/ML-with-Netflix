import numpy as np


class Data:
    def __init__(self):
        file_name_x = '..\\data\\train_ratings_all.dat'
        file_name_y = '..\\data\\train_y_rating.dat'
        file_name_date_x = '..\\data\\train_dates_all.dat'
        file_name_date_y = '..\\data\\train_y_date.dat'
        file_name_titles = '..\\data\\movie_titles.txt'
        self.x = np.loadtxt(file_name_x, dtype='int')
        self.y = np.loadtxt(file_name_y, dtype='int')
        self.dates_x = np.loadtxt(file_name_date_x, dtype='int')
        self.dates_y = np.loadtxt(file_name_date_y, dtype='int')
        titles_file = open(file_name_titles, "r")
        self.titles = [line[5:-1] for line in titles_file]
        # remove newline (empty line) at the end
        self.titles = self.titles[0:99]
        # total data length
        self.n_data = max(self.x.shape)
        self.n_train = []
        self.n_validate = []
        self.n_test = []
        # init split data
        self.x_imputed = []
        self.x_train = []
        self.x_validate = []
        self.x_test = []
        self.y_train = []
        self.y_validate = []
        self.y_test = []

    def manual_split(self, split_data_ratio):
        self.n_train = int(np.round(split_data_ratio[0] * self.n_data))
        self.n_validate = int(np.round(split_data_ratio[1] * self.n_data))
        self.n_test = self.n_data - self.n_validate - self.n_train

        self.x_train = self.x[0:self.n_train, :]
        self.x_validate = self.x[self.n_train:self.n_train+self.n_validate, :]
        self.x_test = self.x[self.n_train+self.n_validate:, :]
        self.y_train = self.y[0:self.n_train]
        self.y_validate = self.y[self.n_train:self.n_train+self.n_validate]
        self.y_test = self.y[self.n_train+self.n_validate:]