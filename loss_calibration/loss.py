import torch


def StepLoss_weighted(weights, threshold):
    """Step Loss

    Args:
        weights (list): cost of misclassification, weights[0] = cost of FN, weights[1]=cost of FP
        threshold (float): threshold for decision-making

    Returns:
        function: loss function L(theta, prediction)
    """

    assert len(weights) == 2, f"Expected 2, got {len(weights)}"

    def loss(true_theta, decision, dim=None):
        """custom loss function (BCE with class weights)

        Args:
            theta (torch.Tensor): observed/true parameter values
            decision (torch.Tensor or float): indicates decision: 0 (below threshold) or  1(above treshold)
            threshold (torch.Tensor): threshold for binarized decisons.

        Returns:
            float: incurred loss
        """

        if type(decision) == float or type(decision) == int:
            assert (
                decision == 0.0 or decision == 1.0
            ), "Decision has to be either 0 or 1"
        else:
            assert torch.logical_or(
                decision == 0.0, decision == 1.0
            ).all(), (
                "All values have to be either 0 (below threshold) or  1(above treshold)"
            )

        return (
            decision
            * (1 - torch.gt(true_theta, threshold).type(torch.float))
            * weights[1]
            + (1 - decision)
            * torch.gt(true_theta, threshold).type(torch.float)
            * weights[0]
        )

    return loss


def SigmoidLoss_weighted(weights, threshold):
    """Sigmoid Loss, differentiable approximation of Step Loss

    Args:
        weights (list): cost of misclassification, weights[0] = cost of FN, weights[1]=cost of FP
        threshold (float): threshold for decision-making

    Returns:
        function: loss function L(theta, prediction)
    """

    assert len(weights) == 2, f"Expected 2, got {len(weights)}"

    def loss(true_theta, decision, dim=None, slope=100):
        """custom loss function (BCE with class weights)

        Args:
            theta (torch.Tensor): observed/true parameter values
            decision (torch.Tensor or float): indicates decision: 0 (below threshold) or  1(above treshold)
            threshold (torch.Tensor): threshold for binarized decisons.

        Returns:
            float: incurred loss
        """

        if type(decision) == float or type(decision) == int:
            assert (
                decision == 0.0 or decision == 1.0
            ), "Decision has to be either 0 or 1"
        else:
            assert torch.logical_or(
                decision == 0.0, decision == 1.0
            ).all(), (
                "All values have to be either 0 (below threshold) or  1(above treshold)"
            )

        return (
            decision
            * (1 - torch.sigmoid(slope * (true_theta - threshold)))
            * weights[1]
            + (1 - decision)
            * torch.sigmoid(slope * (true_theta - threshold))
            * weights[0]
        )

    return loss


def LinearLoss_weighted(weights, threshold):
    """Step Loss

    Args:
        weights (list): cost of misclassification, weights[0] = cost of FN, weights[1]=cost of FP
        threshold (float): threshold for decision-making

    Returns:
        function: loss function L(theta, prediction)
    """

    assert len(weights) == 2, f"Expected 2, got {len(weights)}"

    def loss(true_theta, decision):
        """custom loss function (BCE with class weights)

        Args:
            theta (torch.Tensor): observed/true parameter value
            decision (int): indicates decision: 0 (below threshold) or  1(above treshold)
            threshold (float, optional): threshold for binarized decisons, defaults to 5.0.

        Returns:
            float: incurred loss
        """
        if type(decision) == float or type(decision) == int:
            assert (
                decision == 0.0 or decision == 1.0
            ), "Decision has to be either 0 or 1"
        else:
            assert (
                true_theta.shape == decision.shape
            ), f"Shapes must match, got {true_theta.shape} and {decision.shape}."
            assert torch.logical_or(
                decision == 0.0, decision == 1.0
            ).all(), (
                "All values have to be either 0 (below threshold) or  1(above treshold)"
            )

        return decision * (
            1 - torch.gt(true_theta, threshold).type(torch.float)
        ) * weights[1] * (threshold - true_theta) + (1 - decision) * torch.gt(
            true_theta, threshold
        ).type(
            torch.float
        ) * weights[
            0
        ] * (
            true_theta - threshold
        )

    return loss


# decision                                                0                   1
# * (1 - torch.gt(theta, threshold).type(torch.float))                        th < T
# * weights[1]  # * (1 + (threshold - theta))                                 w1
# + (1 - decision)                                        1
# * torch.gt(theta, threshold).type(torch.float)          th > T
# * weights[0]  # * (1 + (theta - threshold))             w0
#                                                 loss if pred=0, but d_gt = 1, then w0
#                                                                             loss if pred=1, but d_gt=0, then w1
# if w1 >> w0 --> avoid FP, okay with FN
# if w0 >> w1 --> avoid FN, okay with FP


def BCELoss_weighted(weights, threshold, cost_fn="step"):
    def loss(prediction, target, theta):
        """computation of incurred costs

        Args:
            prediction (torch.Tensor): output of the classifier
            target (torch.Tensor): ground truth decision
            theta (torch.Tensor): associated value of theta to compute the cost/weights

        Returns:
            float: loss value
        """
        assert (
            prediction.shape == target.shape == theta.shape
        ), "All arguments should have the same shape."
        assert cost_fn in [
            "step",
            "linear",
        ], "The cost function has to be one of 'step' or 'linear'."

        cost_functions = {
            "step": StepLoss_weighted(weights, threshold),
            "linear": LinearLoss_weighted(weights, threshold),
        }
        costs = cost_functions[cost_fn]

        prediction = torch.clamp(prediction, min=1e-7, max=1 - 1e-7)
        bce = -target * torch.log(prediction) * costs(theta, 0.0) - (
            1 - target
        ) * torch.log(1 - prediction) * costs(theta, 1.0)
        return bce

    return loss
