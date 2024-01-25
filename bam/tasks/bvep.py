import sys
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple

# sys.path.append("./../BVEP/SBI")
import matplotlib.pyplot as plt
import numba
import numpy as np
import torch

# from BVEP_stat_summary import calculate_summary_statistics_features
from sbi.utils import BoxUniform
from sbi.utils.torchutils import atleast_2d
from scipy import signal
from scipy.optimize import fsolve
from scipy.signal import find_peaks, hilbert, savgol_filter
from scipy.stats import kurtosis, moment, skew
from torch import Tensor

from bam.actions import CategoricalAction, UniformAction
from bam.tasks.task import Task


class BVEP(Task):
    def __init__(
        self,
        action_type: str,
        num_actions: int = None,
        probs: List = None,
        simulator="2D",
        rng_seed=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        assert action_type in ["discrete", "continuous"]

        # 6D Epileptor
        if simulator == "6D":
            prior_params = {
                "low": Tensor([-5.0, 2800.0]),
                "high": Tensor([0.0, 3000.0]),
            }
            self.param_low, self.param_high = prior_params["low"], prior_params["high"]
            param_range = {"low": self.param_low, "high": self.param_high}
            self.parameter_names = ["\eta", "\tau_0"]
            dim_data = 800002
            self.simulator = self.simulator6D()

        elif simulator == "2D":
            prior_params = {
                "low": Tensor([-5.0, 0.1, -5.0, 0.0]),
                "high": Tensor([0.0, 50.0, 0.0, 5.0]),
            }
            self.param_low, self.param_high = prior_params["low"], prior_params["high"]
            param_range = {"low": self.param_low, "high": self.param_high}
            self.parameter_names = ["\eta", "\tau", "x_{init}", "z_{init}"]
            dim_data = 4
            self.simulator = self.simulator2Dsummstats()

        else:
            print("Simulator not defined.")
            raise (NotImplementedError)

        parameter_aggregation = lambda params: params
        prior_dist = BoxUniform(**prior_params, device=device.type)

        if action_type == "discrete":
            self.num_actions = num_actions
            self.probs = probs  # ! can be None
            assert num_actions is not None
            actions = CategoricalAction(num_actions=num_actions, probs=probs)
        else:
            raise (NotImplementedError)

        self.rng = np.random.RandomState(seed=rng_seed)
        super().__init__(
            "bvep",
            action_type,
            actions,
            dim_data=dim_data,
            dim_parameters=2,
            prior_params=prior_params,
            prior_dist=prior_dist,
            param_range=param_range,
            parameter_aggregation=parameter_aggregation,
            name_display="Bayesian Virtual Epileptic Patient",
        )

    def simulator2D(self) -> Callable:
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
                x[i] = (
                    x[i - 1] + dt * dx + np.sqrt(dt) * sigma * self.rng.randn()
                )  # np.random.randn()
                z[i] = (
                    z[i - 1] + dt * dz + np.sqrt(dt) * sigma * self.rng.randn()
                )  # np.random.randn()

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

            return states.reshape(-1)

        return Epileptor2Dmodel_simulator_wrapper

    def simulator2Dsummstats(self, return_raw: bool = False) -> Callable:
        def Epileptor2Dmodel(
            params, constants, sigma, dt, ts
        ):  # in BVEP_Simulator.py/VEP2D_forwardmodel, the init conditions seem to be given, not learned as part of the parameters (?)
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
                x[i] = (
                    x[i - 1] + dt * dx + np.sqrt(dt) * sigma * self.rng.randn()
                )  # np.random.randn()
                z[i] = (
                    z[i - 1] + dt * dz + np.sqrt(dt) * sigma * self.rng.randn()
                )  # np.random.randn()

            states = np.concatenate((np.array(x).reshape(-1), np.array(z).reshape(-1)))

            return states

        Epileptor2Dmodel = numba.jit(Epileptor2Dmodel, nopython=False)

        def Epileptor2Dmodel_simulator_wrapper(
            params,
            features=[
                "higher_moments",
                "power_envelope",
                "seizures_onset",
                "spectral_power",
                "amplitude_phase",
            ],
        ):
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

            summstats = torch.as_tensor(
                # self.calculate_summary_statistics(states[0:nt], params, dt, ts, features=features)
                self.calculate_summary_statistics(
                    states[0:nt], ts.shape[0], features=features
                )
            )

            if not return_raw:
                return summstats
            else:
                return summstats, states[0:nt]

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

    def calculate_summary_statistics(
        self,
        x,
        nt,
        features=[
            "higher_moments",
            "power_envelope",
            "seizures_onset",
            "spectral_power",
            "amplitude_phase",
        ],
    ):
        """Calculate summary statistics, copied and adapted from BVEP_stat_summary.py

        Args:
            x (np.array): observation
            nt (int): number of time steps
            features (list): list of strings secifying which features to calculate
        """

        nn = 1  # number of brain regions

        X = x.reshape(-1, nt)  # (ns (=samples), nt (=timesteps))
        ns = X.shape[0]
        n_summary = 100 * nn

        sum_stats_vec = np.concatenate(
            (
                np.mean(X, axis=1),
                np.median(X, axis=1),
                np.std(X, axis=1),
                skew(X, axis=1),
                kurtosis(X, axis=1),
            )
        )

        for item in features:
            if item == "higher_moments":
                sum_stats_vec = np.concatenate(
                    (
                        sum_stats_vec,
                        moment(X, moment=2, axis=1),
                        moment(X, moment=3, axis=1),
                        moment(X, moment=4, axis=1),
                        moment(X, moment=5, axis=1),
                        moment(X, moment=6, axis=1),
                        moment(X, moment=7, axis=1),
                        moment(X, moment=8, axis=1),
                        moment(X, moment=9, axis=1),
                        moment(X, moment=10, axis=1),
                    )
                )

            elif item == "power_envelope":
                X_area = np.trapz(X, dx=0.0001)
                X_pwr = np.sum((X * X), axis=1)
                X_pwr_n = X_pwr / X_pwr.max()

                sum_stats_vec = np.concatenate(
                    (
                        sum_stats_vec,
                        X_area,
                        X_pwr,
                        X_pwr_n,
                    )
                )

            elif item == "seizures_onset":
                seizures_num = []
                seizures_on = []

                for i in np.r_[
                    0:ns
                ]:  # unclear if 0:nn or 0:ns (inconsistent in original code)
                    v = np.zeros(nt)
                    Xhat = np.zeros(nt)
                    Xhat = savgol_filter(X[i, :], 11, 3)

                    v = np.array(Xhat)

                    peaks, _ = find_peaks(v, height=0, rel_height=0.3, width=5)

                    if peaks.shape[0] > 0:
                        seizures_on.append(peaks[0])
                    else:
                        seizures_on.append(100000.0)

                    seizures_num.append(peaks.shape[0])

                sum_stats_vec = np.concatenate(
                    (
                        sum_stats_vec,
                        np.array(seizures_num),
                        np.array(seizures_on),
                    )
                )

            elif item == "spectral_power":
                fs = 10e3

                f, Pxx_den = signal.periodogram(X, fs)

                sum_stats_vec = np.concatenate(
                    (
                        sum_stats_vec,
                        np.max(Pxx_den, axis=1),
                        np.mean(Pxx_den, axis=1),
                        np.median(Pxx_den, axis=1),
                        np.std(Pxx_den, axis=1),
                        skew(Pxx_den, axis=1),
                        kurtosis(Pxx_den, axis=1),
                        moment(Pxx_den, moment=2, axis=1),
                        moment(Pxx_den, moment=3, axis=1),
                        moment(Pxx_den, moment=4, axis=1),
                        moment(Pxx_den, moment=5, axis=1),
                        moment(Pxx_den, moment=6, axis=1),
                        moment(Pxx_den, moment=2, axis=1),
                        moment(Pxx_den, moment=3, axis=1),
                        moment(Pxx_den, moment=4, axis=1),
                        moment(Pxx_den, moment=5, axis=1),
                        moment(Pxx_den, moment=6, axis=1),
                        moment(Pxx_den, moment=7, axis=1),
                        moment(Pxx_den, moment=8, axis=1),
                        moment(Pxx_den, moment=9, axis=1),
                        moment(Pxx_den, moment=10, axis=1),
                        np.diag(np.dot(Pxx_den, Pxx_den.transpose())),
                    )
                )

            elif item == "amplitude_phase":
                analytic_signal = hilbert(X)
                amplitude_envelope = np.abs(analytic_signal)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))

                sum_stats_vec = np.concatenate(
                    (
                        sum_stats_vec,
                        np.mean(amplitude_envelope, axis=1),
                        np.median(amplitude_envelope, axis=1),
                        np.std(amplitude_envelope, axis=1),
                        skew(amplitude_envelope, axis=1),
                        kurtosis(amplitude_envelope, axis=1),
                        np.mean(instantaneous_phase, axis=1),
                        np.median(instantaneous_phase, axis=1),
                        np.std(instantaneous_phase, axis=1),
                        skew(instantaneous_phase, axis=1),
                        kurtosis(instantaneous_phase, axis=1),
                    )
                )

        sum_stats_vec = sum_stats_vec[
            0:n_summary
        ]  # why is this getting shortened here?

        return sum_stats_vec

    def calculate_summary_statistics_notebook(self, x, params, dt, ts, features):
        """Calculate summary statistics copied from Epileptor2D_sde_sbi_fitfeatures.ipynb

        Parameters
        ----------
        x : output of the simulator

        Returns
        -------
        np.array, summary statistics
        """

        params.astype(float)

        n_summary = 100
        nt = ts.shape[0]

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
                # nt = ts.shape[0]
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
                    seizure_times_stim = seizure_times_stim[
                        np.append(1, np.diff(seizure_times_stim)) > 0.75
                    ]

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
            x1, y1, z, x2, y2, u = (
                np.zeros(nt),
                np.zeros(nt),
                np.zeros(nt),
                np.zeros(nt),
                np.zeros(nt),
                np.zeros(nt),
            )
            F1, F2, h = np.zeros(nt), np.zeros(nt), np.zeros(nt)

            states = np.zeros((2 * nt))

            x1_init, y1_init, z_init, x2_init, y2_init, u_init = (
                -2.0,
                -18.0,
                4.0,
                0.0,
                0.0,
                0.0,
            )

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
                dx2 = (
                    -y2[i - 1]
                    + x2[i - 1]
                    - (x2[i - 1] ** 3)
                    + I2
                    + (2 * u[i - 1])
                    - 0.3 * (z[i - 1] - 3.5)
                )
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
        return torch.as_tensor(self.simulator(theta))

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
            print(
                "Some actions are invalid, expected costs with be inf for those actions. "
            )

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
