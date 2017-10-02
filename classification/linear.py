import numpy as np

class LeastSquares:

    def fit(self, X, T):
        # Calculate X transpose
        Xt = np.transpose(X)
        # Calculate pseudo-inverse
        pi = np.linalg(Xt.dot(X)).dot(Xt)
        # Calculate W
        self.W = pi.dot(T)

    def predict(self, X):
        return np.transpose(self.W).dot(X)

class Perceptron:

    def fit(self, X, t, lr=0.001, iterations=5):
        # Calculate number of inputs
        N = X.shape[0]
        # Calculate number of weights
        M = X.shape[1]
        # Initialize weights vector to random values
        self.w = np.random.randint(-1, 2, size=M)
        ## Apply stochasic gradient descent
        for t in range(iterations):
            for i in range(N):
                self.w += lr * X[i] * t[i]

    def predict(self, X):
        # Calculate activation value
        a = np.transpose(X.dot(self.w))
        # Calculate step function on activation values
        y = np.apply_along_axis(class.step_function, 0, a)
        return y

    @staticmethod
    def __step_function__(a):
        return 1 if a[0] >= 0 else -1
