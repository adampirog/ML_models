import numpy as np


class LinearRegressor():
    
    def __init__(self):
        self._theta = None
    
    @staticmethod
    def _normal_equation(X, y):  
        X_b = np.c_[np.ones(X.shape), X]
        
        return np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    @staticmethod
    def _pseudoinverse(X, y):
        # least squares method based on Moore-Penrose inverse
        X_b = np.c_[np.ones(X.shape), X]
        
        return np.linalg.lstsq(X_b, y, rcond=1e-6)[0]
    
    def fit(self, X, y):
        self._theta = self._pseudoinverse(X, y)
        
        return True
    
    def get_theta(self):
        assert self._theta is not None
        
        return self._theta
    
    def predict(self, X):
        assert self._theta is not None
        assert isinstance(X, np.ndarray)
        
        X_new_b = np.c_[np.ones(X.shape), X]
        return X_new_b.dot(self._theta)
    
class GDRegressor():
    
    def __init__(self, max_iter=1_000, tol=0.001, eta0=0.01):
        self._theta = None
        self._max_iter = max_iter
        self._eta = eta0
        self._tol = tol
    
    def fit(self, X, y):
        theta = np.random.randn(2, 1)
        X_b = np.c_[np.ones(X.shape), X]
        m = X.shape[0]
        
        for _ in range(self._max_iter):
            gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
            theta = theta - self._eta * gradients
            
            if np.linalg.norm(gradients) < self._tol:
                break
        
        self._theta = theta
        return True
    
    def get_theta(self):
        assert self._theta is not None
        
        return self._theta
    
    def predict(self, X):
        assert self._theta is not None
        assert isinstance(X, np.ndarray)
        
        X_new_b = np.c_[np.ones(X.shape), X]
        return X_new_b.dot(self._theta)
      
class SGDRegressor():
    
    def __init__(self, max_iter=1_000, tol=0.001, eta0=0.01, power_t=0.25, shuffle=True):
        self._theta = None
        
        self._max_iter = max_iter
        self._eta = self._eta0 = eta0
        self._tol = tol
        self._power_t = power_t
        self._shuffle = shuffle
    
    def _adapt_learning_schedule(self, t):
        power = pow(t, self._power_t)
        
        if power != 0:
            self._eta = self._eta0 / power
            return True
        
        return False
    
    @staticmethod
    def _shuffle_sets(a, b):
        assert len(a) == len(b)
        
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def fit(self, X, y):
        
        if self._shuffle is True:
            X, y = self._shuffle_sets(X, y)
            
        theta = np.random.randn(2, 1)
        X_b = np.c_[np.ones(X.shape), X]
        m = X.shape[0]
        
        for epoch in range(self._max_iter):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X_b[random_index:random_index + 1]
                yi = y[random_index:random_index + 1]
                
                gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                con = self._adapt_learning_schedule(epoch * m + i)

                if con is not True:
                    break
                
                theta = theta - self._eta * gradients
            
            if np.linalg.norm(gradients) < self._tol:
                break
        
        self._theta = theta
        return True
    
    def get_theta(self):
        assert self._theta is not None
        
        return self._theta
    
    def predict(self, X):
        assert self._theta is not None
        assert isinstance(X, np.ndarray)
        
        X_new_b = np.c_[np.ones(X.shape), X]
        return X_new_b.dot(self._theta)
