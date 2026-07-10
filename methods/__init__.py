from methods.base import BasePoseEstimator
from methods.ppf_icp import PPFICPEstimator, PPFICPParams
from methods.ransac import RansacEstimator, RansacParams

def get_estimator(method_name: str, **kwargs) -> BasePoseEstimator:
    """
    Factory function to initialize a 6D Pose Estimator by its name.

    Args:
        method_name (str): Name of the method (e.g. 'ppf_icp', 'ransac').
        **kwargs: Arguments to pass to the method's constructor.

    Returns:
        BasePoseEstimator: The initialized estimator instance.

    Raises:
        ValueError: If the method_name is unrecognized.
        
    Example:
        >>> from methods import get_estimator
        >>> estimator = get_estimator("ransac")
    """
    name_clean = method_name.lower().strip()
    if name_clean in ("ppf", "ppf_icp"):
        return PPFICPEstimator(**kwargs)
    elif name_clean == "ransac":
        return RansacEstimator(**kwargs)
    else:
        raise ValueError(f"Unrecognized pose estimation method name: '{method_name}'")

