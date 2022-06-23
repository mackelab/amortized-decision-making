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

    def loss(theta, decision):
        """custom loss function (BCE with class weights)

        Args:
            theta (torch.Tensor): observed/true parameter value
            decision (int): indicates decision: 0 (below threshold) or  1(above treshold)
            threshold (float, optional): threshold for binarized decisons, defaults to 5.0.

        Returns:
            float: incurred loss
        """
        assert decision in [
            0,
            1,
        ], "decision has to be one of 0 (below threshold) or  1(above treshold)"

        return (
            decision
            * (1 - torch.gt(theta, threshold).type(torch.float))
            * weights[1]  # * (1 + (threshold - theta))
            + (1 - decision)
            * torch.gt(theta, threshold).type(torch.float)
            * weights[0]  # * (1 + (theta - threshold))
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

    def loss(theta, decision):
        """custom loss function (BCE with class weights)

        Args:
            theta (torch.Tensor): observed/true parameter value
            decision (int): indicates decision: 0 (below threshold) or  1(above treshold)
            threshold (float, optional): threshold for binarized decisons, defaults to 5.0.

        Returns:
            float: incurred loss
        """
        assert decision in [
            0,
            1,
        ], "decision has to be one of 0 (below threshold) or  1(above treshold)"

        return decision * (1 - torch.gt(theta, threshold).type(torch.float)) * weights[
            1
        ] * (threshold - theta) + (1 - decision) * torch.gt(theta, threshold).type(
            torch.float
        ) * weights[
            0
        ] * (
            theta - threshold
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


def BCELoss_weighted(weights, threshold, cost_func="step"):
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
        assert cost_func in [
            "step",
            "linear",
        ], "The cost function has to be one of 'step' or 'linear'."

        cost_functions = {
            "step": StepLoss_weighted(weights, threshold),
            "linear": LinearLoss_weighted(weights, threshold),
        }
        costs = cost_functions[cost_func]
        prediction = torch.clamp(prediction, min=1e-7, max=1 - 1e-7)
        bce = -target * torch.log(prediction) * costs(theta, 0) - (
            1 - target
        ) * torch.log(1 - prediction) * costs(theta, 1)
        return bce

    return loss
