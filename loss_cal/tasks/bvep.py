from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numba
import numpy as np
import sbibm
import torch
from sbi.utils import BoxUniform
from sbi.utils.torchutils import atleast_2d
from sbibm.algorithms.sbi.utils import wrap_prior_dist, wrap_simulator_fn
from scipy import stats as spstats
from scipy.optimize import fsolve, root
from scipy.stats import kurtosis, mode, moment, skew
from torch import Tensor
from torch.distributions import Normal

from loss_cal.actions import Action, CategoricalAction, UniformAction
from loss_cal.costs import RevGaussCost, StepCost_weighted
from loss_cal.tasks.task import Task


class BVEP(Task):
    def __init__(
        self,
        action_type: str,
        num_actions: int = None,
        probs: List = None,
    ) -> None:
        assert action_type in ["binary", "continuous"]

        # 6D Epileptor
        # prior_params = {
        #     "low": Tensor([-5.0, 2800.0]),
        #     "high": Tensor([0.0, 3000.0]),
        # }
        # self.param_low, self.param_high = prior_params["low"], prior_params["high"]
        # param_range = {"low": self.param_low, "high": self.param_high}
        # self.parameter_names = ["\eta", "\tau_0"]

        prior_params = {
            "low": Tensor([-5.0, 0.1, -5.0, 0.0]),
            "high": Tensor([0.0, 50.0, 0.0, 5.0]),
        }
        self.param_low, self.param_high = prior_params["low"], prior_params["high"]
        param_range = {"low": self.param_low, "high": self.param_high}
        self.parameter_names = ["\eta", "\tau", "x_{init}", "z_{init}"]

        parameter_aggregation = lambda params: params
        prior_dist = BoxUniform(**prior_params)

        if action_type == "binary":
            self.num_actions = num_actions
            self.probs = probs
            assert num_actions is not None
            actions = CategoricalAction(num_actions=num_actions, probs=probs)
        else:
            raise (NotImplementedError)
            self.action_low, self.action_high = 0.0, 100.0
            actions = UniformAction(low=self.action_low, high=self.action_high, dist=BoxUniform)

        super().__init__(
            "bvep",
            action_type,
            actions,
            dim_data=4,  # 2D full: 2002,  # 6D: 8002,
            dim_parameters=2,
            prior_params=prior_params,
            prior_dist=prior_dist,
            param_range=param_range,
            parameter_aggregation=parameter_aggregation,
            name_display="Bayesian Virtual Epileptic Patient",
        )

    def simulator(self) -> Callable:
        def Epileptor2Dmodel(params, constants, sigma, dt, ts):
            eta, tau, x_init, z_init = params[0], params[1], params[2], params[3]

            eta.astype(float)
            tau.astype(float)
            x_init.astype(float)
            z_init.astype(float)

            # fixed parameters
            I1 = constants[0]
            nt = ts.shape[0]
            dt = float(dt)
            sigma = float(sigma)

            # simulation from initial point
            x = np.zeros(nt)  # fast voltage
            z = np.zeros(nt)  # slow voltage
            states = np.zeros((2, nt))

            x[0] = float(x_init)
            z[0] = float(z_init)

            for i in range(1, nt):
                dx = 1.0 - x[i - 1] ** 3 - 2.0 * x[i - 1] ** 2 - z[i - 1] + I1
                dz = (1.0 / tau) * (4 * (x[i - 1] - eta) - z[i - 1])
                x[i] = x[i - 1] + dt * dx + np.sqrt(dt) * sigma * np.random.randn()
                z[i] = z[i - 1] + dt * dz + np.sqrt(dt) * sigma * np.random.randn()

            states = np.concatenate((np.array(x).reshape(-1), np.array(z).reshape(-1)))

            return states

        Epileptor2Dmodel = numba.jit(Epileptor2Dmodel, nopython=False)

        def Epileptor2Dmodel_simulator_wrapper(params):
            params = np.asarray(params)

            # time step
            T = 100.0
            dt = 0.1
            ts = np.arange(0, T + dt, dt)

            # fixed parameters
            I1 = 3.1
            constants = np.array([I1])

            sigma = 1e-1
            nt = ts.shape[0]

            states = Epileptor2Dmodel(params, constants, sigma, dt, ts)

            summstats = torch.as_tensor(self.calculate_summary_statistics(states[0:nt], params, dt, ts, features=[]))

            return summstats
            # return states.reshape(-1)

        return Epileptor2Dmodel_simulator_wrapper

    def root_fuc(self, roots, eta):
        xx = np.empty(1)
        zz = np.empty(1)
        F = np.empty(2)
        xx = roots[0]
        zz = roots[1]
        I1 = 3.1
        F[0] = 1.0 - xx**3 - 2.0 * xx**2 - zz + I1
        F[1] = 4 * (xx - eta) - zz
        return np.array([F[0], F[1]])

    def calculate_summary_statistics(self, x, params, dt, ts, features):
        """Calculate summary statistics

        Parameters
        ----------
        x : output of the simulator

        Returns
        -------
        np.array, summary statistics
        """

        params.astype(float)

        n_summary = 100

        sum_stats_vec = np.concatenate(
            (
                np.array([np.mean(x)]),
                np.array([np.std(x)]),
                np.array([skew(x)]),
                np.array([kurtosis(x)]),
            )
        )

        for item in features:
            if item == "higher_moments":
                sum_stats_vec = np.concatenate(
                    (
                        sum_stats_vec,
                        np.array([moment(x, moment=2)]),
                        np.array([moment(x, moment=3)]),
                        np.array([moment(x, moment=4)]),
                        np.array([moment(x, moment=5)]),
                        np.array([moment(x, moment=6)]),
                        np.array([moment(x, moment=7)]),
                        np.array([moment(x, moment=8)]),
                        np.array([moment(x, moment=9)]),
                        np.array([moment(x, moment=10)]),
                    )
                )

            if item == "seizures_onset":
                # initialise array of seizure counts
                nt = ts.shape[0]
                v = np.zeros(nt)
                v = np.array(x)

                v_th = 0
                ind = np.where(v < v_th)
                v[ind] = v_th

                ind = np.where(np.diff(v) < 0)
                v[ind] = v_th

                seizure_times = np.array(ts)[ind]
                seizure_times_stim = seizure_times

                if seizure_times_stim.shape[0] > 0:
                    seizure_times_stim = seizure_times_stim[np.append(1, np.diff(seizure_times_stim)) > 0.75]

                sum_stats_vec = np.concatenate(
                    (
                        sum_stats_vec,
                        np.array([seizure_times_stim.shape[0]]),
                    )
                )

            if item == "fixed_point":
                rGuess = np.array([[-1.0, 3.0]])
                true_roots = fsolve(self.root_fuc, rGuess, args=(params[0],))

                sum_stats_vec = np.concatenate(
                    (
                        sum_stats_vec,
                        true_roots,
                    )
                )

        sum_stats_vec = sum_stats_vec[0:n_summary]

        return sum_stats_vec

    def simulator6D(self) -> Callable:
        def Epileptor6Dmodel(params, constants, dt, ts):
            eta = params[0]
            tau1 = params[1]
            eta.astype(float)
            tau1.astype(float)

            # fixed parameters
            tau2, I1, I2, gamma = constants[0], constants[1], constants[2], constants[3]

            x1th = 0
            x2th = -0.25

            nt = ts.shape[0]
            dt = float(dt)

            # simulation from initial point
            x1, y1, z, x2, y2, u = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
            F1, F2, h = np.zeros(nt), np.zeros(nt), np.zeros(nt)

            states = np.zeros((2 * nt))

            x1_init, y1_init, z_init, x2_init, y2_init, u_init = -2.0, -18.0, 4.0, 0.0, 0.0, 0.0

            x1[0] = float(x1_init)
            y1[0] = float(y1_init)
            z[0] = float(z_init)
            x2[0] = float(x2_init)
            y2[0] = float(y2_init)
            u[0] = float(u_init)

            sigma1, sigma, sigma2 = 0.0, 0.0, 0.0

            for i in range(1, nt):
                if x1[i - 1] < 0:
                    F1[i - 1] = (x1[i - 1] ** 3) - 3 * (x1[i - 1] ** 2)
                else:
                    F1[i - 1] = (x2[i - 1] - (0.6 * ((z[i - 1] - 4) ** 2))) * x1[i - 1]

                if x2[i - 1] < -0.25:
                    F2[i - 1] = 0
                else:
                    F2[i - 1] = 6 * (x2[i - 1] + 0.25) * x1[i - 1]

                h[i - 1] = 4 * (x1[i - 1] - eta)
                # h=(eta+3.0)/(1+exp((-x1-0.5)/0.1));

                dx1 = y1[i - 1] - F1[i - 1] - z[i - 1] + I1
                dy1 = 1.0 - (5 * x1[i - 1] ** 2) - y1[i - 1]
                dz = (1 / tau1) * (h[i - 1] - z[i - 1])
                dx2 = -y2[i - 1] + x2[i - 1] - (x2[i - 1] ** 3) + I2 + (2 * u[i - 1]) - 0.3 * (z[i - 1] - 3.5)
                dy2 = (1 / tau2) * (-y2[i - 1] + F2[i - 1])
                du = -gamma * (u[i - 1] - 0.1 * x1[i - 1])

                x1[i] = x1[i - 1] + dt * dx1 + np.sqrt(dt) * sigma1 * np.random.randn()
                y1[i] = y1[i - 1] + dt * dy1 + np.sqrt(dt) * sigma1 * np.random.randn()
                z[i] = z[i - 1] + dt * dz + np.sqrt(dt) * sigma * np.random.randn()
                x2[i] = x2[i - 1] + dt * dx2 + np.sqrt(dt) * sigma2 * np.random.randn()
                y2[i] = y2[i - 1] + dt * dy2 + np.sqrt(dt) * sigma2 * np.random.randn()

            states = np.concatenate((np.array(x1).reshape(-1), np.array(z).reshape(-1)))

            return states

        Epileptor6Dmodel = numba.jit(Epileptor6Dmodel, nopython=False)

        def Epileptor6Dmodel_simulator_wrapper(params):
            params = np.asarray(params)

            # time step
            T = 4000.0
            dt = 0.01
            ts = np.arange(0, T + dt, dt)

            # fixed parameters
            tau2 = 10.0
            I1 = 3.1
            I2 = 0.45
            gamma = 0.01

            constants = np.array([tau2, I1, I2, gamma])

            nt = ts.shape[0]

            states = Epileptor6Dmodel(params, constants, dt, ts)

            return states.reshape(-1)

        return Epileptor6Dmodel_simulator_wrapper

    def get_prior(self) -> Callable[..., Any]:
        return self.prior_dist

    def get_simulator(self) -> Callable[..., Any]:
        return self.sample_simulator

    def sample_prior(self, n: int) -> Tensor:
        return self.prior_dist.sample((n,))

    def sample_simulator(self, theta: Tensor) -> Tensor:
        return self.simulator()(theta)

    def evaluate_prior(self, theta: Tensor) -> Tensor:
        return self.prior_dist.log_prob(theta)

    def expected_posterior_costs(
        self,
        x: int or Tensor,
        a: Tensor,
        cost_fn: Callable[..., Any],
        param: int,
        verbose=True,
    ) -> Tensor:
        # make sure tensors are 2D
        a = atleast_2d(a)
        if verbose and not (self.actions.is_valid(a)).all():
            print("Some actions are invalid, expected costs with be inf for those actions. ")

        expected_costs = torch.empty_like(a)
        mask = self.actions.is_valid(a)
        expected_costs[:, torch.logical_not(mask)] = torch.inf

        a_valid = a[:, mask]
        raise (NotImplementedError)
        post = None
        if param is not None:
            post = post[:, param : param + 1]
            incurred_costs = cost_fn(post, a_valid)
        else:
            incurred_costs = cost_fn(self.param_aggregation(post), a_valid)
        # expected posterior costs
        expected_costs[:, mask] = incurred_costs.mean(dim=0)
        return expected_costs

    ## TODO
    def bayes_optimal_action(
        self,
        x_o: Tensor,
        a_grid: Tensor,
        cost_fn: Callable,
        lower: float = 0.0,
        upper: float = 5.0,
        resolution: int = 500,
    ) -> float:
        """Compute the Bayes optimal action under the ground truth posterior

        Args:
            x_o (Tensor): observation, conditional of the posterior p(theta|x=x_o)
            a_grid (Tensor): actions to compute the incurred costs for
            lower (float, optional): lower bound the parameter grid/integral. Defaults to 0.0.
            upper (float, optional): upper bound of the parameter grid/integral. Defaults to 5.0.
            resolution (int, optional): number of evaluation points. Defaults to 500.
            cost_fn (Callable, optional): cost function to compute incurred costs. Defaults to RevGaussCost(factor=1).

        Returns:
            float: action with minimal incurred costs
        """
        raise NotImplementedError
        losses = torch.tensor(
            [
                self.expected_posterior_costs(
                    x=x_o, a=a, lower=lower, upper=upper, resolution=resolution, cost_fn=cost_fn
                )
                for a in a_grid
            ]
        )
        return a_grid[losses.argmin()]

    def bayes_optimal_action_binary(
        self,
        n: int,
        param: int,
        cost_fn: Callable = StepCost_weighted(weights=[5.0, 1.0], threshold=2.0),
        verbose=False,
    ) -> float:
        """Compute the Bayes optimal action under the ground truth posterior for binary action

        Args:
            x_o (Tensor): observation, conditional of the posterior p(theta|x=x_o)
            a_grid (Tensor): actions to compute the incurred costs for
            lower (float, optional): lower bound the parameter grid/integral. Defaults to 0.0.
            upper (float, optional): upper bound of the parameter grid/integral. Defaults to 5.0.
            resolution (int, optional): number of evaluation points. Defaults to 500.
            cost_fn (Callable, optional): cost function to compute incurred costs. Defaults to StepCost_weighted(weights=[5.0, 1.0], threshold=2.0).

        Returns:
            float: action with minimal incurred costs
        """
        costs_action0, costs_action1 = self.expected_posterior_costs(
            n=n, a=Tensor([[0.0], [1.0]]), param=param, cost_fn=cost_fn, verbose=verbose
        )
        return (costs_action0 > costs_action1).float()

    def posterior_ratio_binary(
        self,
        n: int,
        param: int,
        cost_fn: Callable = StepCost_weighted(weights=[5.0, 1.0], threshold=2.0),
        verbose=False,
    ) -> float:
        """Compute the posterior ratio: (exp. costs taking action 0)/(exp. costs taking action 0 + exp. costs taking action 1)

        Args:
            x_o (Tensor): observation, conditional of posterior p(theta|x_o)
            lower (float, optional): lower bound of the parameter grid/integral. Defaults to 0.0.
            upper (float, optional): upper bound of the parameter grid/inetgral. Defaults to 5.0.
            resolution (int, optional): number of evaluation points. Defaults to 500.
            cost_fn (Callable, optional): cost function to compute incurred costs.Defaults to StepCost_weighted(weights=[5.0, 1.0], threshold=2.0).
        Returns:
            float: posterior ratio
        """
        int_0, int_1 = self.expected_posterior_costs(
            n=n, a=Tensor([[0.0], [1.0]]), param=param, cost_fn=cost_fn, verbose=verbose
        )
        return int_0 / (int_0 + int_1)
