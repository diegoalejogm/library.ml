import numpy as np

### Discriminative Functions ###

# Binary classification [X]
# Multi-class classification [ ]
class LeastSquares:

    def fit(self, X, T):
        # Calculate X transpose
        Xt = np.transpose(X)
        # Calculate pseudo-inverse
        pi = np.linalg.inv(Xt.dot(X)).dot(Xt)
        # Calculate W
        self.W = pi.dot(T)

    def predict(self, X):
        return np.transpose(self.W).dot(X)

# Binary classification [X]
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

### Probabilistic Generative Models ###

# Binary classification [X]
# Multi-class classification [ ]
class MaximumLikelihood:

    def fit(X, t):
        # Calculate N: number of inputs x in X
        N = X.shape[0]
        # Obtain elements that belong to class 1
        X1 = numpy.take(X, [i for i in range(N) if t[i][0] == 1], axis=0)
        # Obtain elements that belong to c0: class 0
        X0 = numpy.take(X, [i for i in range(N) if t[i][0] == 0], axis=0)
        # Calculate N1: num. inputs x in X that belong to class 1
        N1 = X1.shape[0]
        # Calculate N0: num. inputs x in X that belong to class 0
        N0 = X0.shape[0]
        # Calculate pi : p(C1)
        pi = N1 / (N1 + N0)
        # Calculate miu1: mean of elements in class 1
        m1 = np.mean(X1, axis=0)
        # Calculate miu0: mean of elements in class 0
        m1 = np.mean(X0, axis=0)
        # Calculate S1 : Variance of X1
        S1 = np.transpose(X1).dot(X1)
        # Calculate S0 : Variance of X0
        S0 = np.transpose(X0).dot(X0)
        # Calculate S : Variance of X (also weighted sum of S1 and S0)
        S = S1 * (N1/ (N1+N0)) + S0 * (N0/ (N1+N0))

        ##  GAUSSIAN PARAMETERS
        # Calculate Epsilon
        self.epsilon = S
        # Calculate weights
        self.w = np.linalg.inv(epsilon).dot(np.subtract(m1, m0))
        # Calculate bias
        self.b =  -1/2 * np.transpose(m1).dot(np.linalg.inv(epsilon)).dot(m1)
        self.b +=  -1/2 * np.transpose(m0).dot(np.linalg.inv(epsilon)).dot(m0)
        self.b += np.log(pi/(1-pi))

    def predict(X):
        return np.transpose(self.w).dot(X).sum(self.b)
