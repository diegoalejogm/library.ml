import numpy as np

# Maximum Likelihood (Frequentist)
class LeastSquaresLinearRegression:

    def __init__(self,  phi=None, learning_rate=None):
        self.phi = phi
        if learning_rate:
            self.learning_rate = learning_rate

    def fit(self, X, Y):
        if hasattr(self,'learning_rate'):
            self.__fit_sequential__(X, Y)
        else:
            self.__fit_non_sequential__(X, Y)

    def predict(self, T):
        assert self.w != None or self.phi!= None, "Model has not been trained."
        T = self.__as_2d_array__(T)
        phi_T = np.array([self.__apply_basis_function__(t_i) for t_i in T])
        return np.dot(phi_T, self.w)

    def __fit_sequential__(self, x, t):
        # Reshape x and t as 2D arrays
        x = self.__as_2d_array__(x)
        t = self.__as_2d_array__(t)

        # Calculate number of weights
        self.M = x.shape[1] + 1

        # Create dummy basis functions phi_i(x) = x if none was given
        if self.phi is None:
            self.phi = [(lambda x: x[i]) for i in range(self.M-1)]
            # Add bias basis function phi_i(x) = 1
            self.phi.insert(0, lambda x: 1)

        # Initialize weights for the first time
        if not hasattr(self, 'w'):
            weights_shape = (self.M, 1)
            self.w = np.zeros(weights_shape)
        # Calculate phi for current x
        phi_x = self.__apply_basis_function__(x)
        phi_x = self.__as_2d_array__(phi_x)
        # Calculate prediction y = transpose(w) * phi_x
        prediction = np.dot(np.transpose(self.w), phi_x)
        # Calculate error for current iteration
        error = np.subtract(t, prediction)
        # Calculate gradient for current iteration
        error_gradient = error * phi_x
        # Update weights
        self.w = np.add(self.w, self.learning_rate * error_gradient)

    def __fit_non_sequential__(self, X, Y):
        # Number of rows in X
        self.N = X.shape[0]

        # Convert X and Y into matrices if it is vector for homogeneous treatment.
        X = self.__as_2d_array__(X)
        Y = self.__as_2d_array__(Y)

        # Number of weights
        self.M = X.shape[1] + 1

        # Create dummy basis functions phi_i(x) = x if none was given
        if self.phi is None:
            self.phi = [(lambda x: x[i]) for i in range(self.M-1)]
        # Add bias basis function phi_i(x) = 1
        self.phi.insert(0, lambda x: 1)

        # Calculate design matrix (NxM)
        PHI = np.array([self.__apply_basis_function__(x_i) for x_i in X])
        # Calculate Moore-Penrose Pseudo Inverse
        mp_pseudoinv = np.linalg.pinv(PHI)
        self.w = np.dot(mp_pseudoinv, Y)


    # Helper Methods
    def __apply_basis_function__(self, x):
        ans = np.array([ self.phi[j](x) for j in range(self.M)])
        return ans

    def __as_2d_array__(self,a):
        if len(a.shape) == 0:
            a = np.array([a])
        if len(a.shape) == 1:
            A = np.reshape(a, (a.shape[0], 1))
            return A
        else:
            return a

# Correct solution
sol = np.array([[25.], [205.], [-53.], [31.], [33.]])
sol_weights = np.array([5,2])

# Non sequential learning test
my_X = np.array([1, 2, 3])
my_y = 5 + my_X * 2
my_t = np.array([10, 100, -29, 13, 14])
model = LeastSquaresLinearRegression()
model.fit(my_X, my_y)
p = model.predict(my_t)
np.testing.assert_array_almost_equal(sol, p)
print('Predictions Non Sequential:\n{}'.format(p))

# Sequential learning test
model = LeastSquaresLinearRegression(learning_rate=0.001)
my_X = np.arange(0,5,0.01)
my_y = 5 + my_X * 2
for _ in range(100):
    for i in range(len(my_X)):
        model.fit(my_X[i], my_y[i])
my_t = np.array([10, 100, -29, 13, 14])
p = np.rint(model.predict(my_t))
np.testing.assert_array_almost_equal(sol, p)
print('Predictions Sequential:\n{}'.format(p))
