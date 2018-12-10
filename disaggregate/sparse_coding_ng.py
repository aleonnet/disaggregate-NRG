import os
from os.path import join
import pandas as pd
import numpy as np

from sklearn.decomposition import SparseCoder
from sklearn.decomposition import sparse_encode
from sklearn.decomposition import DictionaryLearning

import matplotlib.pyplot as plt

from const import REDD_DIR, TRAIN_END

house_id = 1
path = os.path.join(REDD_DIR, 'building_{0}.csv'.format(house_id))


class SparseCoding(object):
    def __init__(self, n, transform_algorithm='lars'):
        self.n = n
        self.net = DictionaryLearning(n_components=n, alpha=0.8, max_iter=1000)
        self.net.set_params(transform_algorithm=transform_algorithm)


    def plot_B(self, B):
        plt.figure(figsize=(4.2, 4))
        for i, comp in enumerate(B[:self.n]):
            plt.subplot(10, 10, i + 1)
            plt.imshow(comp, cmap=plt.cm.gray_r,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())

        plt.suptitle('Dictionary learned from time series\n' +
                     'Train time %.1fs on %d patches' % (dt, len(data)),
                     fontsize=16)

        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


    def init_weights(self, X_mat):
        B, A , recon = [], [], []
        for app in X_mat:
            data = X_mat[app].reshape(1, -1)
            B_i = self.net.fit(data).components_
            A_i = self.net.transform(data)
            X_hat = np.dot(A_i, B_i)

            B.append(B_i)
            A.append(A_i)
            recon.append(X_hat)

            print("MSE Error: ", np.mean((data - X_hat) ** 2))

        return A, B, recon




        # transform_algorithms = [
        #     ('Orthogonal Matching Pursuit\n1 atom', 'omp',
        #      {'transform_n_nonzero_coefs': 1}),
        #     ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
        #      {'transform_n_nonzero_coefs': 2}),
        #     ('Least-angle regression\n5 atoms', 'lars',
        # ]
        #
        # reconstructions = {}
        # for title, transform_algorithm, kwargs in transform_algorithms:
        #     print(title + '...')
        #     reconstructions[title] = .copy()
        #     t0 = time()
        #     dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
        #     code = dico.transform(data)
        #     patches = np.dot(code, V)
        #
        #     patches += intercept
        #     patches = patches.reshape(len(data), *patch_size)
        #     if transform_algorithm == 'threshold':
        #         patches -= patches.min()
        #         patches /= patches.max()
        #     reconstructions[title][:, width // 2:] = reconstruct_from_patches_2d(
        #         patches, (height, width // 2))
        #     dt = time() - t0
        #     print('done in %.2fs.' % dt)
        #     show_with_diff(reconstructions[title], face,
        #                    title + ' (time: %.1fs)' % dt)
        #
        # plt.show()


def discr_training(app_matrix, A, B):
    pass

def train(app_matrix, net):
    A, B, recon = net.init_weights(app_matrix)





house_data = pd.read_csv(path)
house_data = house_data.set_index(pd.DatetimeIndex(house_data['time'])).drop('time', axis=1)

apps = house_data.columns.values
apps = apps[apps != 'Main']
train_data = house_data[:TRAIN_END]
dev_data = house_data[TRAIN_END:]

net = SparseCoding(n = 10)
app_matrix = { app : train_data[app].values for app in apps }

train(app_matrix, net)






# X_train, y_train = train_data.drop('main', axis=1), train_data.main
# X_dev, y_dev = dev_data.drop('main', axis=1), dev_data.main
#
# sc = SparseCoding()
# sc.fit(X_train, y_train)
