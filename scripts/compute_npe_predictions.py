import torch
import glob
from os import path
from loss_calibration.loss import BCELoss_weighted, StepLoss_weighted


def post_ratio(post, x_o, loss, lower=0.0, upper=5.0, resolution=500):
    thetas = torch.linspace(lower, upper, resolution).unsqueeze(dim=1)
    probs = post.log_prob(thetas, x=x_o).exp().unsqueeze(dim=1)
    assert thetas.shape == probs.shape
    # expected posterior loss
    loss_fn = (probs * loss(thetas, 0) * (upper - lower) / resolution).sum()
    loss_fp = (probs * loss(thetas, 1) * (upper - lower) / resolution).sum()
    return loss_fn / (loss_fn + loss_fp)


if __name__ == "__main__":
    threshold = 2.0
    costs = [5.0, 1.0]
    step_loss = StepLoss_weighted(costs, threshold)

    # load test data
    th_test = torch.load(path.join("../data/1d_classifier/", "th_test.pt"))
    x_test = torch.load(path.join("../data/1d_classifier/", "x_test.pt"))
    d_test = (th_test > threshold).float()
    N_test = th_test.shape[0]
    print("N_test = ", N_test)

    # load sbi posterior
    flow = "nsf"
    files = sorted(glob.glob(path.join("../results/sbi/", f"2022-05*{flow}*0.pt")))
    print(f"Generating plots for:")
    # files = [files[-2]] + files[:-2]
    posteriors = []
    nsim = []

    for file in files:
        print("- ", file)
        posteriors.append(torch.load(file))
        nsim.append(int(file.split("_")[-1].split(".")[0]))

    for i, p in enumerate(posteriors):
        preds = torch.as_tensor(
            [post_ratio(p, x_o, step_loss) for x_o in x_test]
        ).unsqueeze(dim=1)
        torch.save(
            preds,
            path.join(
                "./results/sbi/",
                f"{files[i].split('/')[-1].split('.')[0]}_predictions_t{int(threshold)}_c{int(costs[0])}_{int(costs[1])}.pt",
            ),
        )
        print(i, end=",")
