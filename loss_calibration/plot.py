import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import path
from loss_calibration.utils import raw_stats_given_predictions

# def plot_decision_boundaries(
#     clfs,
#     test_data,
#     xlim=(10, 210),
#     threshold
# ):
#     x_linspace = torch.linspace(xlim[0], xlim[1], 1000).unsqueeze(dim=1)
#     color_cycle = cycler(color=plt.cm.coolwarm(np.linspace(0, 1, len(clfs))))

#     decision_boundaries = []
#     with mpl.rc_context(fname="loss_calibration/.matplotlibrc"):
#         plt.scatter(th_test, x_test, s=2)
#         for clf, c in zip(clfs, color_cycle):
#             preds_linspace = clf(x_linspace)
#             idx = (torch.abs(preds_linspace - 0.5)).argmin()
#             plt.axhline(
#                 x_linspace[idx], **c, label=", ".join(map(str, clf._summary["weights"]))
#             )
#             decision_boundaries.append(x_linspace[idx])
#         plt.axvline(threshold, c="k", label="threshold")
#         plt.xlabel(r"$\theta_{eval}$")
#         plt.ylabel(r"$x_{eval}$")
#         # plt.title('decision boundary for varying costs of misclassification')

#         plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
#         plt.savefig("results/1d_classifier/1D_clf_decision_boundaries.pdf")
#         plt.show()


def plot_loss(
    loss,
    low,
    high,
    threshold,
    figsize=(4, 2),
    labels=[r"$L(\theta, 0)$", r"$L(\theta, 1)$"],
    labels_backgorund=[r"$D_1$", r"$D_2$"],
    xlabel=r"$\theta$",
    ylabel=r"$p(\theta)$",
    resolution=100,
    save=False,
    save_path="results/classifier/concept/",
):
    with mpl.rc_context(fname="loss_calibration/.matplotlibrc"):

        thetas = torch.arange(low, high, (high - low) / resolution)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.axvline(x=threshold, color="k", label="threshold")
        ax.axvspan(
            threshold, high * 1.05, facecolor="r", alpha=0.1, label=labels_backgorund[0]
        )
        ax.axvspan(
            low * 1.05, threshold, facecolor="y", alpha=0.1, label=labels_backgorund[1]
        )

        plt.plot(thetas, loss(thetas, 0), color="#EFB913", label=labels[0])
        plt.plot(thetas, loss(thetas, 1), color="#EC400E", label=labels[1])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(low, high)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=6,
        )
        if save:
            plt.savefig(path.join(save_path, "costs.pdf"))
        plt.show()


def plot_predictions(
    predictions,
    theta_test,
    x_test,
    threshold,
    figsize=(4, 2),
    marker_size=5,
):
    tp, fn, fp, tn = raw_stats_given_predictions(predictions, theta_test, threshold)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(
        theta_test[tp], x_test[tp], c="#3D7D28", marker="+", s=marker_size, label="TP"
    )
    ax.scatter(
        theta_test[tn],
        x_test[tn],
        edgecolors="#4D9436",
        marker="o",
        s=marker_size,
        facecolors="none",
        label="TN",
    )
    ax.scatter(
        theta_test[fn], x_test[fn], c="#BF1150", marker="+", s=marker_size, label="FN"
    )
    ax.scatter(
        theta_test[fp],
        x_test[fp],
        edgecolors="#E55E1E",
        marker="o",
        s=marker_size,
        facecolors="none",
        label="FP",
    )
    ax.axvline(threshold, c="k")
    ax.set_title("Predictions")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$x$")
    ax.legend()
    return fig, ax


def plot_stats_vs_n():
    pass


def heatmap_joint(
    ax,
    values,
    cmap="viridis",
    extent=[0, 5, 10, 220],
    vmin=0,
    vmax=1,
    norm=None,
    return_im=False,
):
    if norm is None:
        im = ax.imshow(
            values.rot90(),
            cmap=cmap,
            interpolation="none",
            extent=extent,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
    else:
        im = ax.imshow(
            values.rot90(),
            cmap=cmap,
            interpolation="none",
            extent=extent,
            aspect="auto",
            norm=norm,
        )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$x$")

    if return_im:
        return ax, im
    else:
        return ax
