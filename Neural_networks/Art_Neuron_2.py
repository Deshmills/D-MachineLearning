import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self.unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        '''[n_samples/number of rows, n_features/number of columns]'''
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])
        # y_ = converting all the values in the target/Y into 0 or to 1

        for _ in range(self.n_iters):
            # enumerate to iterate over the training samples/rows idx for index, and xi for sample
            for idx, xi in enumerate(X):
                # ------------------------
                linear_output = np.dot(xi, self.weights)+self.bias
                y_predicted = self.activation_func(linear_output)
                # Calculating the predicted value ( Ŷ ) for xi / training sample 'This is similar to the [def predict(X,y)] function but this is for each training sample.
# -----------------------

# -----------------------

                ''' Perceptron rule update
                ΔW [delta Weight / Update weight]: = α = [learning rate] * (yi [target] - Ŷ [prediction]) * xi

                ΔW: = α * (yi - Ŷ ) * xi
                '''

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * xi
                self.bias += update

    def predict(self, X):
        '''Approximation:

        Ŷ = Activation function(WᵗX + bias) = 

        '''
        linear_output = np.dot(X, self.weights) + self.bias
        # Linear output = (WᵗX + bias)
        y_predicted = self.activation_func(linear_output)
        return y_predicted  # Ŷ  = Y prediction

    def unit_step_function(self, X):
        '''Unit step function or activation function'''
        return np.where(X >= 0, 1, 0)
