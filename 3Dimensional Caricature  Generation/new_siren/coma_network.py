import scipy
import numpy as np
import torch
# import graph


class coma():
    def __init__(self, MADU):
        super(coma, self).__init__()
        # Dict = np.load(opt.MADU, allow_pickle=True)
        Dict = np.load(MADU, allow_pickle=True)
        self.L = Dict.item()['L']
        self.p = Dict.item()['p']
        self.U = Dict.item()['U']
        self.F = [16, 16, 16, 32]  # Number of graph convolutional filters.
        self.K = [6, 6, 6, 6]  # Polynomial orders.
        self.F_0 = 6  # Number of graph input features.
        self.regularizers = []

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = torch.matmul(x, W) + b
        return torch.nn.ReLU(x) if relu else x

    def unpool(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)

        x = x.permute([1, 2, 0])  # M x Fin x N
        x = x.reshape([M, Fin*N])  # M x Fin*N
        x = tf.sparse_tensor_dense_matmul(L, x) # Mp x Fin*N
        x = x.reshape([Mp, Fin, N])  # Mp x Fin x N
        x = x.permute([2,0,1]) # N x Mp x Fin

        return x

    def filter(self, x, L, Fout, K):  # chebyshev5
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        # L = graph.rescale_L(L, lmax=2)
        L = rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = x.permute([1, 2, 0])  # M x Fin x N
        x0 = x0.reshape([M, Fin*N])  # M x Fin*N
        x = torch.unsqueeze(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = torch.unsqueeze(x_, 0)  # 1 x M x Fin*N
            return torch.cat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = x.reshape([K, M, Fin, N])  # K x M x Fin x N
        x = x.permute([3,1,2,0])  # N x M x Fin x K
        x = x.reshape([N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = torch.matmul(x, W)  # N*M x Fout
        return x.reshape([N, M, Fout])  # N x M x Fout

    def brelu(self, x):  # b1relu
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return torch.nn.ReLU(x + b)

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def decode(self, x, reuse=tf.AUTO_REUSE, name = 'decoder'):
        #with tf.variable_scope('fc-encoder'):
         #   x = self.fc(x, int(x.shape[1]))

        with tf.variable_scope(name, reuse=reuse):
            N = x.get_shape()[0]
            with tf.variable_scope('fc2'):
                x = self.fc(x, int(self.p[-1]*self.F[-1]))

            x = x.reshape([int(N), int(self.p[-1]), int(self.F[-1])])

            for i in range(len(self.F)):
                with tf.variable_scope('upconv{}'.format(i+1)):
                    with tf.name_scope('unpooling'):
                        x = self.unpool(x, self.U[-i-1])
                    with tf.name_scope('filter'):
                        x = self.filter(x, self.L[len(self.F)-i-1], self.F[-i-1], self.K[-i-1])
                    with tf.name_scope('bn'):
                        x = tf.layers.batch_normalization(x)
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)
            with tf.name_scope('outputs'):
                #x = self.filter(x, self.L[0], int(self.F_0), self.K[0])  # (16, 6144, 6)
                x = self.filter(x, self.L[0], 3, self.K[0])  # (16, 6144, 6)
        return x


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L
