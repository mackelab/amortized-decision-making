from copy import deepcopy
from os import path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from sbi.inference import SNLE, SNPE, MCMCPosterior, RejectionPosterior, VIPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.utils import mcmc_transform
from sbi.utils.torchutils import atleast_2d
from torch import zeros

import loss_calibration.linear_gaussian as lin_gauss
import loss_calibration.lotka_volterra as lv
import loss_calibration.toy_example as toy
from loss_calibration.loss import SigmoidLoss_weighted, StepLoss_weighted


class AcquisitionPotential(BasePotential):
    allow_iid_x = False

    def __init__(self, prior, x_o, acquisition_fn, device="cpu"):
        super().__init__(prior, x_o, device=device)
        self.acquisition_fn = acquisition_fn

    def __call__(self, theta, track_gradients=True):
        with torch.set_grad_enabled(track_gradients):
            return torch.log(self.acquisition_fn(theta))


class Acquisition:
    def __init__(
        self,
        threshold,
        threshold_dim,
        costs,
        likelihood_estimator,
        posterior_estimator,
        num_monte_carlo_samples_likelihood=5,
        num_monte_carlo_samples_posterior=1,
    ):
        self.threshold = threshold
        self.threshold_dim = threshold_dim
        self.cost_fn = SigmoidLoss_weighted(weights=costs, threshold=threshold)
        self.likelihood_estimator = likelihood_estimator
        self.posterior_estimator = posterior_estimator
        self.num_monte_carlo_samples_likelihood = num_monte_carlo_samples_likelihood
        self.num_monte_carlo_samples_posterior = num_monte_carlo_samples_posterior

    def update_estimators(self, likelihood_estimator, posterior_estimator):
        self.likelihood_estimator = likelihood_estimator
        self.posterior_estimator = posterior_estimator

    def check_estimators(self):
        assert (
            self.likelihood_estimator is not None
            and self.posterior_estimator is not None
        ), "Set estimators first."

    def __call__(self, theta_0):
        self.check_estimators()
        theta_0 = atleast_2d(theta_0)
        # 1.sample from likelihood
        predicted_x = self.likelihood_estimator.sample(
            self.num_monte_carlo_samples_likelihood, context=theta_0
        )
        predicted_theta_given_x = []
        # 2. for each sampled x, sample from the posterior
        for px in predicted_x.swapaxes(0, 1):
            predicted_theta_given_x.append(
                self.posterior_estimator.sample(
                    self.num_monte_carlo_samples_posterior, context=px
                )
            )
        predicted_theta_given_x = torch.cat(predicted_theta_given_x, dim=1)
        # 3. calculate associated cost
        predicted_decision = (predicted_theta_given_x > self.threshold).float()
        estimated_cost = self.cost_fn(
            theta_0[:, self.threshold_dim]
            .unsqueeze(1)
            .repeat(1, self.num_monte_carlo_samples_likelihood),
            predicted_decision[:, :, self.threshold_dim],
        )
        return estimated_cost.mean(dim=1)


class ActiveLearning:
    def __init__(
        self,
        prior,
        simulator,
        threshold: float,
        threshold_dim: int,
        costs: list,
        num_monte_carlo_samples_likelihood: int = 20,
        num_monte_carlo_samples_posterior: int = 1,
        density_estimator: str = "nsf",
        save_dir: str = "./",
    ) -> None:
        self.prior = prior
        self.simulator = simulator
        self.proposal = prior
        self.potential = None

        self.num_monte_carlo_samples_likelihood = num_monte_carlo_samples_likelihood
        self.num_monte_carlo_samples_posterior = num_monte_carlo_samples_posterior

        self.costs = costs
        self.threshold = threshold
        self.threshold_dim = threshold_dim
        self.cost_fn = SigmoidLoss_weighted(costs, threshold)

        self.inference_likelihood = SNLE(prior, density_estimator=density_estimator)
        self.inference_posterior = SNPE(prior, density_estimator=density_estimator)
        self.likelihood_estimator = None
        self.posterior_estimator = None

        self.n_per_round = 1000
        self.theta_roundwise = []

        self.save_dir = save_dir

    def acquisition_fn(self, theta_0):
        theta_0 = atleast_2d(theta_0)
        # 1.sample from likelihood
        predicted_x = self.likelihood_estimator.sample(
            self.num_monte_carlo_samples_likelihood, context=theta_0
        )
        predicted_theta_given_x = []
        # 2. for each sampled x, sample from the posterior
        for px in predicted_x.swapaxes(0, 1):
            predicted_theta_given_x.append(
                self.posterior_estimator.sample(
                    self.num_monte_carlo_samples_posterior, context=px
                )
            )
        predicted_theta_given_x = torch.cat(predicted_theta_given_x, dim=1)
        # 3. calculate associated cost
        predicted_decision = (predicted_theta_given_x > self.threshold).float()
        estimated_cost = self.cost_fn(
            theta_0[:, self.threshold_dim]
            .unsqueeze(1)
            .repeat(1, self.num_monte_carlo_samples_likelihood),
            predicted_decision[:, :, self.threshold_dim],
        )

        return estimated_cost.mean(dim=1)

    def run(self, n_rounds, sample="rejection"):
        assert sample in [
            "rejection",
            "mcmc",
        ], "Only rejection sampling and MCMC sampling currently supported."

        for r in range(n_rounds):
            print(f"\n----- ROUND {r} -----")
            # sample theta from proposal and simulate
            theta = self.proposal.sample((self.n_per_round,))
            self.theta_roundwise.append(theta)
            x = self.simulator(theta)

            # train NLE + NPE on all data
            self.likelihood_estimator = self.inference_likelihood.append_simulations(
                theta, x
            ).train(max_num_epochs=100)
            self.posterior_estimator = self.inference_posterior.append_simulations(
                theta, x, proposal=self.proposal
            ).train(max_num_epochs=100)

            self.potential = AcquisitionPotential(
                self.prior,
                x_o=zeros(1, x.shape[1]),
                acquisition_fn=self.acquisition_fn,
                device="cpu",
            )

            if sample == "mcmc":
                prior_tf = mcmc_transform(self.prior)
                acquisition_sampler = MCMCPosterior(
                    potential_fn=self.potential,
                    theta_transform=prior_tf,
                    proposal=self.prior,
                    init_strategy="proposal",
                    method="slice_np_vectorized",
                    num_chains=100,
                )
            else:
                acquisition_sampler = RejectionPosterior(
                    potential_fn=self.potential, proposal=self.prior
                )
            self.proposal = acquisition_sampler

            self.save_round(r)

        return (
            self.likelihood_estimator,
            self.posterior_estimator,
            self.potential,
            self.proposal,
        )

    def save_round(self, round):
        torch.save(
            torch.stack(self.theta_roundwise),
            path.join(self.save_dir, f"sampled_theta.pt"),
        )
        torch.save(
            self.likelihood_estimator,
            path.join(self.save_dir, f"likelihood_estimator_round{round}.pt"),
        )
        torch.save(
            self.posterior_estimator,
            path.join(self.save_dir, f"posterior_estimator_round{round}.pt"),
        )
        torch.save(
            self.proposal,
            path.join(self.save_dir, f"proposal_round{round}.pt"),
        )


@hydra.main(
    version_base=None,
    config_path="../scripts/configs/",
    config_name="active_learning_config",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    _ = torch.manual_seed(cfg.seed)

    if cfg.task.name == "lotka_volterra":
        prior = lv.get_prior()
        sim = lv.get_simulator()
    elif cfg.task.name == "toy_example":
        prior = toy.get_prior()
        sim = toy.get_simulator()
    elif cfg.task.name == "linear_gaussian":
        prior = lin_gauss.get_prior()
        sim = lin_gauss.get_simulator()
    else:
        raise ValueError("Task not defined.")

    threshold = round(cfg.task.T, ndigits=4)
    parameter = cfg.task.parameter
    costs = list(cfg.task.costs)

    num_monte_carlo_samples_likelihood = cfg.num_monte_carlo_samples_likelihood
    num_monte_carlo_samples_posterior = cfg.num_monte_carlo_samples_posterior

    acquisition_fn = Acquisition(
        threshold=threshold,
        threshold_dim=parameter,
        costs=costs,
        likelihood_estimator=None,
        posterior_estimator=None,
        num_monte_carlo_samples_likelihood=num_monte_carlo_samples_likelihood,
        num_monte_carlo_samples_posterior=num_monte_carlo_samples_posterior,
    )

    n_rounds = cfg.n_rounds
    proposal = prior

    # inference objects
    inference_likelihood = SNLE(prior, density_estimator=cfg.density_estimator)
    inference_posterior = SNPE(prior, density_estimator=cfg.density_estimator)

    for r in range(n_rounds):  # rounds
        print(f"\n----- ROUND {r} -----")
        theta = proposal.sample((1000,))
        x = sim(theta)

        # train both SNLE and SNPE
        likelihood_estimator = inference_likelihood.append_simulations(theta, x).train(
            max_num_epochs=100
        )
        posterior_estimator = inference_posterior.append_simulations(
            theta, x, proposal=proposal
        ).train(max_num_epochs=100)

        acquisition_fn.update_estimators(
            deepcopy(likelihood_estimator), deepcopy(posterior_estimator)
        )
        potential = AcquisitionPotential(
            prior, x_o=zeros(1, 2), acquisition_fn=acquisition_fn, device="cpu"
        )
        if cfg.sampler == "mcmc":
            prior_tf = mcmc_transform(prior)
            acquisition_sampler = MCMCPosterior(
                potential_fn=potential,
                theta_transform=prior_tf,
                proposal=prior,
                init_strategy="proposal",  # "sir"
                method="slice_np_vectorized",
                num_chains=100,
            )
        else:
            acquisition_sampler = RejectionPosterior(
                potential_fn=potential, proposal=prior
            )
        proposal = acquisition_sampler

        threshold_to_str = str(threshold).replace(".", "_")
        costs_to_str = str(int(costs[0])) + "_" + str(int(costs[1]))

        torch.save(
            theta,
            path.join(
                cfg.res_dir,
                "active_learning",
                cfg.task.name,
                f"round{r}_sampled_theta.pt",
            ),
        )
        torch.save(
            potential,
            path.join(
                cfg.res_dir,
                "active_learning",
                cfg.task.name,
                f"t{parameter}_{threshold_to_str}_c{costs_to_str}_round{r}_potential.pt",
            ),
        )

        torch.save(
            proposal,
            path.join(
                cfg.res_dir,
                "active_learning",
                cfg.task.name,
                f"t{parameter}_{threshold_to_str}_c{costs_to_str}_round{r}_proposal.pt",
            ),
        )
        torch.save(
            likelihood_estimator,
            path.join(
                cfg.res_dir,
                "active_learning",
                cfg.task.name,
                f"t{parameter}_{threshold_to_str}_c{costs_to_str}_round{r}_likelihood_estimator.pt",
            ),
        )
        torch.save(
            posterior_estimator,
            path.join(
                cfg.res_dir,
                "active_learning",
                cfg.task.name,
                f"t{parameter}_{threshold_to_str}_c{costs_to_str}_round{r}_posterior_estimator.pt",
            ),
        )


if __name__ == "__main__":
    main()
