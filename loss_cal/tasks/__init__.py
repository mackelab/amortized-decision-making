from typing import Any
import torch


def get_task(task_name: str, action_type="continuous", **kwargs: Any):
    """Provide an instance of a task

    Args:
        task_name (str): name of task, one of ['toy_example', 'lotka_volterra', 'sir', 'linear_gaussian', 'bvep']
        action_type (str, optional): type of action, one of ['continuous', 'discrete']. Defaults to "continuous".

    Raises:
        NotImplementedError: raise error for tasks that are not implemented yet

    Returns:
        Task: task object
    """
    assert action_type in [
        "discrete",
        "continuous",
    ], "action_type have to be either 'continuous' or 'discrete'."

    if task_name == "toy_example":
        from loss_cal.tasks.toy_example import ToyExample

        return ToyExample(action_type=action_type, **kwargs)

    elif task_name == "lotka_volterra":
        from loss_cal.tasks.lotka_volterra import LotkaVolterra

        return LotkaVolterra(action_type=action_type, **kwargs)

    if task_name == "sir":
        from loss_cal.tasks.sir import SIR

        return SIR(action_type=action_type, **kwargs)

    if task_name == "linear_gaussian":
        from loss_cal.tasks.linear_gaussian import LinGauss

        return LinGauss(action_type=action_type, **kwargs)

    if task_name == "bvep":
        from loss_cal.tasks.bvep import BVEP

        return BVEP(action_type=action_type, **kwargs)

    else:
        raise NotImplementedError()


def get_task_specs(task_name):
    task_dict = {
        "toy_example": {
            "action_type": "continuous",
            "param": 0,
            "factor": 2,
            "exponential": 2,
            "aligned": True,
            "offset": 0.0,
            "lr": 1e-3,
        },
        "sir": {
            "action_type": "continuous",
            "param": None,
            "factor": 2,
            "exponential": 2,
            "aligned": False,
            "offset": 1.0,
            "lr": 1e-3,
        },
        "lotka_volterra": {
            "action_type": "continuous",
            "param": "?",
            "factor": 3,
            "exponential": 2,
            "aligned": True,
            "offset": 0.0,
            "lr": 0.005,
        },
        "linear_gaussian": {
            "action_type": "continuous",
            "param": None,
            "factor": 0.5,
            "exponential": 2,
            "aligned": True,
            "offset": torch.abs(get_task("linear_gaussian").param_high[0]).item(),
            "lr": 1e-3,
        },
        "bvep": {
            "action_type": "discrete",
            "param": None,
            "num_actions": 3,
            "thresholds": "[-3.05,-2.05]",
            "transform": "Sigmoid",
            "lr": 1e-3,
        },
    }

    assert (
        task_name in task_dict.keys()
    ), f"Provided task must be one of {list(task_dict.keys())}"

    return task_dict[task_name]
