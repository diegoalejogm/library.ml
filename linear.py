import numpy as np

# Maximum Likelihood (Frequentist)
class LinearRegression:

    def fit(self, X, Y, phi=None):
        # Number of rows in X
        self.N = X.shape[0]

        # Convert X and Y into matrices if it is vector for homogeneous treatment.
        if len(X.shape) == 1:
            X = self.__as_2d_matrix__(X)
        if len(Y.shape) == 1:
            Y = self.__as_2d_matrix__(Y)

        # Number of weights
        self.M = X.shape[1] + 1

        # Create dummy basis function phi_i(x) = x if none was given
        if phi is None:
            phi = [(lambda x: x[i]) for i in range(self.M-1)]

        # Add bias basis function phi_i(x) = 1
        phi.insert(0, lambda x: 1)
        self.phi = phi

        # Calculate design matrix (NxM)
        PHI = np.array([self.__apply_basis_function__(x_i) for x_i in X])
        # Calculate Moore-Penrose Pseudo Inverse
        mp_pseudoinv = np.linalg.pinv(PHI)
        self.w = np.dot(mp_pseudoinv, Y)

    def predict(self, T):
        assert self.w != None or self.phi!= None, "Model has not been trained."
        T = self.__as_2d_matrix__(T)
        phi_T = np.array([self.__apply_basis_function__(t_i) for t_i in T])
        return np.dot(phi_T, self.w)

    # Helper Methods
    def __apply_basis_function__(self, x):
        ans = np.array([ self.phi[j](x) for j in range(self.M)])
        return ans

    def __as_2d_matrix__(self,a):
        if len(a.shape) == 1:
            A = np.reshape(a, (a.shape[0], 1))
        return A


# Quick test
my_X = np.array([1, 2, 3])
my_y = np.array([6, 7, 8])
my_t = np.array([10, 100, -29, 13, 14])
model = LinearRegression()
model.fit(my_X, my_y)
p = model.predict(my_t)
