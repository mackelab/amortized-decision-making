from typing import Any


def get_task(task_name: str, *args: Any, **kwargs: Any):
    """Get task

    Args:
        task_name: Name of task

    Returns:
        Task instance
    """

    if task_name == "toy_example":
        from loss_cal.tasks.toy_example import ToyExample

        return ToyExample(*args, **kwargs)

    elif task_name == "lotka_volterra":
        from loss_cal.tasks.lotka_volterra import LotkaVolterra

        return LotkaVolterra(*args, **kwargs)

    if task_name == "sir":
        from loss_cal.tasks.sir import SIR

        return SIR(*args, **kwargs)

    elif task_name == "linear_gaussian":
        from loss_cal.tasks.linear_gaussian import LinGauss

        return LinGauss(*args, **kwargs)

    else:
        raise NotImplementedError()
