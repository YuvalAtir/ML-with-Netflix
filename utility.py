import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import interactive


class Data:
    def __init__(self):
        file_name_x = 'C:\\Users\\Yuval\\Google Drive\\MachineLearning\\Netflix\\data\\train_ratings_all.dat'
        file_name_y = 'C:\\Users\\Yuval\\Google Drive\\MachineLearning\\Netflix\\data\\train_y_rating.dat'
        file_name_date = 'C:\\Users\\Yuval\\Google Drive\\MachineLearning\\Netflix\\data\\train_y_date.dat'
        file_name_titles = 'C:\\Users\\Yuval\\Google Drive\\MachineLearning\\Netflix\\data\\movie_titles.txt'
        self.x = np.loadtxt(file_name_x, dtype='int')
        self.y = np.loadtxt(file_name_y, dtype='int')
        self.dates = np.loadtxt(file_name_date, dtype='int')
        titles_file = open(file_name_titles, "r")
        self.titles = [line[5:-1] for line in titles_file]
        self.titles = self.titles[0:99]   # remove newline (empty line) at the end
        self.x_nonzero = []

    def count_zeros(self):
        # count non zeros in data
        zero_count = (self.x == 0).sum()
        zero_count1 = self.x.shape[0]*self.x.shape[1] - np.count_nonzero(self.x, axis=1)
        print(zero_count,zero_count1)
        # count non zeros in data per column
        col_zero_count = (self.x == 0).sum(0)
        plt.plot(col_zero_count, '.')
        plt.xlabel('#movie')
        plt.ylabel('number of missing values')
        plt.title('num of zeros in each column')
        plt.show()  # waits for user to close the window to continue

    def replace_zeros(self,replace_val):
        self.x_nonzero = self.x
        self.x_nonzero[self.x == 0] = replace_val


class SomeCorrelations:
    def __init__(self, x=0, y=0, my_data_frame=[]):
        # correlation between two vectors
        self.corr = np.correlate(x, y)
        # correlation coefficient [2,2] matrix [xx, xy, yx, yy]
        self.coef = np.corrcoef(x, y)
        # pearsonr returns also the 2-tail p-value
        [self.coefp, self.pval] = scipy.stats.stats.pearsonr(x, y)
        print(self.corr, '\n', self.coef, '\n', self.coefp, '\n', self.pval)
        # graph correlation coefficient
        if my_data_frame:  # not empty
            data_frame_corr = my_data_frame.data_frame.corr(method='pearson')
        else:
            data_frame_corr = np.zeros((5,5))
        plt.imshow(data_frame_corr, interpolation='nearest')
        plt.grid(True)
        plt.colorbar()
        plt.show()


class MyDataFrame:
    def __init__(self,data):
        self.data_frame = pd.DataFrame(data.x[:, 0:14], columns=data.titles[0:14])
        self.data_frame['y'] = data.y

    def statistics(self,data):
        # prints the data with titles
        print(self.data_frame.describe())
        # The average rating of the first 14 movies (that have no NANs)
        print(self.data_frame.mean())

        plt.plot(data.dates, data.y, '.')
        plt.xlabel('date')
        plt.ylabel('rating')
        plt.show()

        plt.hist(tr_y, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        plt.show()

    @staticmethod
    def append_y(self, data):
        # append the Data.y as a column of the matrix Data.x
        # need to change y size form (10000,) to (10000,1)
        y_reshape = data.y.reshape(10000, 1)
        x_append = np.append(data.x, y_reshape, axis=1)
        return x_append

def main():
    nf_data = Data()
    #interactive(True)  # draw plot as soon as created
                        # requires to add: raw_input('press return to continue')

    mdf = MyDataFrame(nf_data)
    SomeCorrelations(nf_data.x[:,0], nf_data.y, mdf)

    nf_data.count_zeros()
    replace_val = 1
    nf_data.replace_zeros(replace_val)
    nf_data.count_zeros()

    mdf = MyDataFrame(nf_data)
    mdf.statistics(nf_data)


if __name__ == "__main__": main()
