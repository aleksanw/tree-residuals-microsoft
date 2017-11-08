import sklearn
import numpy as np

from utils import assert_shapetype


class Approximator_Table:
    def __init__(self, action_space):
        self.action_space = action_space
        self.table = {}

    def learn(self, learning_rate, X, Y_target):
        """Update approximator, correcting `learning_rate` of the error.

        Parameters
        ----------
        learning_rate : weight of 
        X : sequence of vectors -- features
        Y_target : sequence of scalars -- target values corresponding to features in X
        """

        Y = self(X)
        Y_target = np.asarray(Y_target)
        Y_update = learning_rate*Y_target + (1-learning_rate)*Y

        self.table.update(zip(map(tuple, X), Y_update))

    def __call__(self, X):
        """Evaluate approximator at each x in X.

        Parameters
        ----------
        X : sequence of scalars or vectors -- features

        Returns
        -------
        numpy array of approximated values
        """
        # Coerce scalars to 1-dim vectors
        X = np.reshape(X, (-1,1))
        return np.fromiter((self.table.get(tuple(x), 0.0) for x in X),
                            np.float64, count=len(X))


class Approximator_ResidualBoosting:
    """Gradient boosted trees approximator.
    Features may be vectors or scalars.  Value is scalar.
    TODO: Require features to be vectors.  Makes for easier debugging.
    """
    def __init__(self, action_space):
        self.action_space = action_space
        self.approximators = []
        self.learning_rates = []

    def learn(self, learning_rate, X, Y_target):
        """Update approximator, correcting `learning_rate` of the error.

        Parameters
        ----------
        learning_rate : weight of 
        X : sequence of scalars or vectors -- features
        Y_target : sequence of scalars -- target values corresponding to features in X
        """
        assert_shapetype(X, 'int64', (-1,-1))
        assert_shapetype(Y_target, 'float64', (-1,1))

        # This function is not pure at all.  The returned tree is owned by the
        # Regressor and will be in-place replaced by future calls to fit.
        # Instantiate a new Regressor for every fiting.
        fit_tree = sklearn.tree.DecisionTreeRegressor(max_depth=2).fit

        X = np.asarray(X)
        Y_target = np.asarray(Y_target)

        Y_residual = Y_target - self(X)
        h = fit_tree(X, Y_residual).predict

        # As in Microsoft's paper, apply learning_rate after fitting.
        #     h = lambda X: learning_rate*h(X)
        # Avoid expensive lambdas by instead applying learning rate on
        # evaluation.  Save learning_rate for this purpose.
        self.learning_rates.append(learning_rate)

        # Sidenote: It should be more or less equivalent to apply the learning
        # rate to the residuals before fitting, saving on storage and computation.

        self.approximators.append(h)

    def __call__(self, X):
        """Evaluate approximator at each x in X.

        Parameters
        ----------
        X : sequence of scalars or vectors -- features

        Returns
        -------
        numpy array of approximated values
        """
        assert_shapetype(X, 'int64', (-1,-1))
        # Approximators do not yet have learning rates applied.  Do that during
        # summation.
        sum_ = np.zeros((len(X),1))
        for lr, h in zip(self.learning_rates, self.approximators):
            Y = h(X).reshape(-1,1)
            sum_ += lr * Y

        assert_shapetype(sum_, 'float64', (-1,1))
        return sum_
