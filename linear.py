import numpy as np

# Maximum Likelihood
class LinearRegression:

    def fit(self, X, y):
        N = X.shape[0]
        bias = np.ones(N)
        X = np.column_stack((bias,X))
        Xt = np.transpose(X)
        mp_pseudoinv = np.linalg.pinv(np.dot(Xt, X))
        self.w = np.dot(np.dot(mp_pseudoinv, Xt), y)

    def predict(self, t):
        N = t.shape[0]
        bias = np.ones(N)
        t = np.column_stack((bias,t))
        assert self.w != None, "Model is not trained"
        return np.dot(t, self.w)


## Quick sample
# my_X = np.array([1, 2, 3])
# my_y = np.array([6, 7, 8])
# my_t = np.array([10, 100, -29, 13, 14])
#
# model = LinearRegression()
# model.fit(my_X, my_y)
# p = model.predict(my_t)
# print(p)
