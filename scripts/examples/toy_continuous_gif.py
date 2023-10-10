import imageio
import matplotlib.pyplot as plt
import torch
from sbi.utils.torchutils import atleast_2d

from loss_cal.costs import RevGaussCost
from loss_cal.predictor import FeedforwardNN
from loss_cal.tasks.toy_example import ToyExample

toy = ToyExample()
device = "cpu"
nn = FeedforwardNN(
    input_dim=2,
    hidden_dims=[20, 20, 20],
    output_dim=1,
    mean=torch.Tensor([102.3523, 2.5074]),  # TODO: not hardcoded!
    std=torch.Tensor([48.1500, 1.4457]),
    activation=torch.nn.ReLU(),
)
model_ckp = torch.load("./results/toy_example/continuous/notebook/best_model.pt", map_location=torch.device(device))
nn.load_state_dict(model_ckp["state_dict"])


obs = torch.arange(30, 210, 10)
a_grid = torch.arange(0.0, 5.0, 0.05)

lower, upper = 0.0, 5.0
resolution = 500
theta_grid = torch.linspace(lower, upper, resolution).unsqueeze(1)


def create_frame(x_o, i):
    fig = plt.figure(figsize=(10, 3))

    ## obtain posterior and expected costs
    post = toy.gt_posterior(x_o, lower, upper, resolution)
    expected_posterior_losses = torch.tensor(
        [
            toy.expected_posterior_costs(
                x=x_o, a=a, lower=lower, upper=upper, resolution=resolution, cost_fn=RevGaussCost(factor=1)
            )
            for a in a_grid
        ]
    )

    predicted_losses = torch.tensor([nn(atleast_2d(x_o), atleast_2d(a)) for a in a_grid])

    # plot
    plt.subplot(1, 2, 1)
    plt.plot(theta_grid, post, label="post")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$p(\theta|x)$")
    plt.title("posterior")
    # plt.title(rf'$x_o=${x_o}')
    plt.xlim(-0.1, 5.1)
    plt.ylim(-0.01, 3)

    plt.subplot(1, 2, 2)
    plt.plot(a_grid, expected_posterior_losses, label="true")
    plt.plot(a_grid, predicted_losses, label="NN")
    plt.xlabel(r"action $a$")
    plt.ylabel("expected costs")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("expected costs")
    # plt.title(rf'$x_o=${x_o}')
    plt.ylim(0.1, 1.1)
    plt.xlim(-0.1, 5.1)

    plt.suptitle(rf"$x_o=${x_o}")
    plt.tight_layout()
    plt.savefig(f"./results/toy_example/continuous/expected_costs_{i}.png", transparent=False, facecolor="white")
    plt.close()


if __name__ == "__main__":
    for i, x_o in enumerate(obs):
        create_frame(x_o, i)

    frames = []
    for i, x_o in enumerate(obs):
        image = imageio.v2.imread(f"./results/toy_example/continuous/expected_costs_{i}.png")
        frames.append(image)
    imageio.mimsave(
        "./results/toy_example/continuous/expected_costs.gif",  # output gif
        frames,  # array of input frames
        duration=500,  # optional: frames per second
    )
