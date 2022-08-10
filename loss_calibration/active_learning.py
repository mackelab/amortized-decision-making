import torch
from torch import zeros

from sbi.inference import SNLE, SNPE
from sbi.utils.torchutils import atleast_2d

from sbi.utils import mcmc_transform
from sbi.inference import MCMCPosterior, RejectionPosterior
from sbi.inference.potentials.base_potential import BasePotential

from copy import deepcopy

from loss_calibration.loss import SigmoidLoss_weighted
from os import path


class AcquisitionPotential(BasePotential):
    allow_iid_x = False

    def __init__(self, prior, x_o, acquisition_fn, device="cpu"):
        super().__init__(prior, x_o, device=device)
        self.acquisition_fn = acquisition_fn

    def __call__(self, theta, track_gradients=True):
        with torch.set_grad_enabled(track_gradients):
            return torch.log(self.acquisition_fn(theta))


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

        self.inference_likelihood = SNLE(prior)
        self.inference_posterior = SNPE(prior)
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
        # print(self.proposal)
        # torch.save(
        #     self.proposal,
        #     path.join(self.save_dir, f"proposal_round{round}.pt"),
        # )
