import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import random
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)
# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self, **kwargs):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        prop_defaults = {
            "xfv": np.empty((0, 3)),
            "var_f": 0.15**2,
            "var_v": 0.0001**2,
            "penalty": 5.0,
            "margin": 3.3,
            "beta": 1.0,
        }

        self.__dict__.update(prop_defaults)
        self.__dict__.update(kwargs)

        self.gp_f = GaussianProcessRegressor(
            kernel=0.5 * Matern(length_scale_bounds="fixed", length_scale=0.5, nu=2.5),
            alpha=self.var_f,
            optimizer=None,
            normalize_y=False,
            random_state=42)

        self.gp_v = GaussianProcessRegressor(
            kernel=ConstantKernel(1.5) + ((2)**0.5) * Matern(length_scale_bounds="fixed", length_scale=0.5, nu=2.5),
            alpha=self.var_v,
            optimizer=None,
            normalize_y=False,
            random_state=42)

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        
        x = self.xfv[:, 0][:, np.newaxis]
        f = self.xfv[:, 1][:, np.newaxis]
        v = self.xfv[:, 2][:, np.newaxis]
        
        self.gp_f.fit(X=x, y=f)
        self.gp_v.fit(X=x, y=v)
        
        return (self.optimize_acquisition_function())

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below.

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        # UCB acquisition function

        mu_f, std_f = self.gp_f.predict(x, return_std=True)
        mu_v, std_v = self.gp_v.predict(x, return_std=True)

        return mu_f[0] + self.beta*std_f[0]-self.penalty*max(mu_v[0]+self.margin*std_v[0]-SAFETY_THRESHOLD, 0)

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.

        self.xfv = np.vstack((self.xfv, np.array([[float(x), float(f), float(v)]])))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.

        x, f, v = self.xfv[:, 0], self.xfv[:, 1], self.xfv[:, 2]
        mask = v < SAFETY_THRESHOLD
        x, f, v = x[mask], f[mask], v[mask]
        return x[f.argmax()]
        

# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])

def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)

def v(x: float):
    """Dummy SA"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init

def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
