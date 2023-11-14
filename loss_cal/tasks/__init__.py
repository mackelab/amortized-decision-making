from typing import Any


def get_task(task_name: str, action_type="continuous", **kwargs: Any):
    """Get task

    Args:
        task_name: Name of task

    Returns:
        Task instance
    """
    assert action_type in ["discrete", "continuous"], "action_type have to be either 'continuous' or 'discrete'."
    print("task_name = ", task_name)

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
