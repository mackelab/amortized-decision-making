from abc import abstractmethod
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sbibm
import torch
from sbi.utils.torchutils import atleast_2d
from sbibm.algorithms.sbi.utils import wrap_prior_dist, wrap_simulator_fn
from torch import Tensor

from loss_cal.actions import Action


class Task:
    def __init__(
        self,
        name: str,
        action_type: str,
        actions: Action,
        dim_data: int,
        dim_parameters: int,
        prior_params: Dict[str, Tensor],
        prior_dist: Callable,
        param_range: Dict[str, Tensor],
        parameter_aggregation: Callable,
        name_display: Optional[str] = None,
    ) -> None:
        assert action_type in ["binary", "continuous"]

        self.task_name = name
        self.display_name = name_display if name_display is not None else name

        self.dim_data = dim_data
        self.dim_parameters = dim_parameters
        self.prior_dist = prior_dist
        self.prior_params = prior_params
        self.param_range = param_range
        self.param_aggregation = parameter_aggregation
        self.action_type = action_type
        self.actions = actions

    @abstractmethod
    def get_prior(self) -> Callable:
        """Get function returning parameters from prior"""
        raise NotImplementedError

    def get_prior_dist(self) -> torch.distributions.Distribution:
        """Get prior distribution"""
        return self.prior_dist

    @abstractmethod
    def get_simulator(self) -> Callable:
        """Get function returning observations from simulator"""
        raise NotImplementedError

    @abstractmethod
    def sample_simulator(self, theta: Tensor) -> Tensor:
        """Get observations from simulator

        Args:
            theta (Tensor): parameter to simulate

        Returns:
            Tensor: observations
        """
        raise NotImplementedError

    @abstractmethod
    def sample_prior(self, n: int) -> Tensor:
        """Get paramters from prior
        Args:
            n (int): number of samples

        Returns:
            Tensor: samples from prior
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_prior(self, theta: Tensor) -> Tensor:
        "evaluate log prob of a parameter"
        raise NotImplementedError

    @abstractmethod
    def expected_posterior_costs(
        self,
        x: int or Tensor,
        a: Tensor,
        cost_fn: Callable,
        param: int or None,
        verbose=True,
    ) -> Tensor:
        """Compute expected costs under the posterior"""
        raise NotImplementedError

    def generate_data(
        self,
        ntrain: int,
        ntest: int,
        nval: int,
        base_dir="./data",
        save_data=True,
        show_progress=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        dir = path.join(base_dir, self.task_name)

        n = ntrain + ntest + nval
        thetas = self.sample_prior(n=n)
        if show_progress:
            xs = torch.empty((n, self.dim_data))
            simulator = self.get_simulator()
            for i, th in enumerate(thetas):
                xs[i, :] = simulator(theta=th)
                print(f"{i}/{n}", end="\r")
        else:
            xs = self.sample_simulator(theta=thetas)

        theta_train = thetas[:ntrain]
        x_train = xs[:ntrain]

        theta_val = thetas[ntrain : ntrain + nval]
        x_val = xs[ntrain : ntrain + nval]

        theta_test = thetas[ntrain + nval :]
        x_test = xs[ntrain + nval :]

        if save_data:
            torch.save(theta_train, path.join(dir, "theta_train.pt"))
            torch.save(x_train, path.join(dir, "x_train.pt"))
            torch.save(theta_val, path.join(dir, "theta_val.pt"))
            torch.save(x_val, path.join(dir, "x_val.pt"))
            torch.save(theta_test, path.join(dir, "theta_test.pt"))
            torch.save(x_test, path.join(dir, "x_test.pt"))
            print(f"Generated new training, test and vailadation data. Saved at: {dir}")

        return theta_train, x_train, theta_val, x_val, theta_test, x_test


class BenchmarkTask(Task):
    def __init__(
        self,
        name: str,
        action_type: str,
        actions: Action,
        param_range: Dict[str, Tensor],
        parameter_aggegration: Callable,
    ) -> None:
        self._task = sbibm.get_task(name)
        super().__init__(
            name=name,
            action_type=action_type,
            actions=actions,
            dim_data=self._task.dim_data,
            dim_parameters=self._task.dim_parameters,
            prior_params=self._task.get_prior_params(),
            prior_dist=self._task.get_prior_dist(),
            param_range=param_range,
            parameter_aggregation=parameter_aggegration,
            name_display=self._task.name_display,
        )

        self.num_observations = self._task.num_observations
        self.parameter_names = self._task.get_labels_parameters()

    def get_prior(self) -> Callable[..., Any]:
        return self._task.get_prior()

    def get_simulator(self) -> Callable[..., Any]:
        return self._task.get_simulator()

    def sample_prior(self, n: int) -> Tensor:
        prior = self.get_prior()
        return prior(num_samples=n)

    def sample_simulator(self, theta: Tensor) -> Callable[..., Any]:
        simulator = self.get_simulator()
        return simulator(theta)

    def evaluate_prior(self, theta: Tensor):
        return self.prior_dist.log_prob(theta)

    def get_observation(self, n: int):
        return self._task.get_observation(num_observation=n)

    def get_reference_samples(self, n: int):
        return self._task.get_reference_posterior_samples(num_observation=n)

    def get_true_parameters(self, n: int):
        return self._task.get_true_parameters(num_observation=n)

    def expected_posterior_costs(
        self, x: int, a: Tensor, cost_fn: Callable[..., Any], param: int, verbose=True
    ) -> Tensor:
        assert type(x) == int, "Provide index of the reference observation."
        a = atleast_2d(a)
        if verbose and not (self.actions.is_valid(a)).all():
            print("Some actions are invalid, expected costs with be inf for those actions. ")

        post = self.get_reference_samples(n=x)

        expected_costs = torch.empty_like(a)
        mask = self.actions.is_valid(a)
        a_valid = a[:, mask]
        expected_costs[:, torch.logical_not(mask)] = torch.inf

        if param is not None:  # restrict posterior samples to one parameter if given
            post = post[:, param : param + 1]
            incurred_costs = cost_fn(post, a_valid)
        else:
            incurred_costs = cost_fn(self.param_aggregation(post), a_valid)
        expected_costs[:, mask] = incurred_costs.mean(dim=0)
        return expected_costs

    def generate_data(
        self,
        ntrain: int,
        ntest: int,
        nval: int,
        base_dir="./data",
        automatic_transforms_enabled: bool = False,  ##
        save_data=True,
        show_progress=False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        dir = path.join(base_dir, self.task_name)

        prior = self.get_prior_dist()
        simulator = self.get_simulator()

        transforms = self._task._get_transforms(automatic_transforms_enabled)["parameters"]
        if automatic_transforms_enabled:
            prior = wrap_prior_dist(prior, transforms)
            simulator = wrap_simulator_fn(simulator, transforms)

        n = ntrain + ntest + nval
        thetas = prior((n,))
        if show_progress:
            xs = torch.empty((n, self.dim_data))
            for i, th in enumerate(thetas):
                xs[i, :] = simulator(th)
                print(f"{i}/{n}", end="\r")
        else:
            xs = simulator(thetas)

        theta_train = thetas[:ntrain]
        x_train = xs[:ntrain]

        theta_val = thetas[ntrain : ntrain + nval]
        x_val = xs[ntrain : ntrain + nval]

        theta_test = thetas[ntrain + nval :]
        x_test = xs[ntrain + nval :]

        if save_data:
            torch.save(theta_train, path.join(dir, "theta_train.pt"))
            torch.save(x_train, path.join(dir, "x_train.pt"))
            torch.save(theta_val, path.join(dir, "theta_val.pt"))
            torch.save(x_val, path.join(dir, "x_val.pt"))
            torch.save(theta_test, path.join(dir, "theta_test.pt"))
            torch.save(x_test, path.join(dir, "x_test.pt"))
            print(f"Generated new training, test and vailadation data. Saved at: {dir}")

        return theta_train, x_train, theta_val, x_val, theta_test, x_test
