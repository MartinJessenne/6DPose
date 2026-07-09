from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d

class BasePoseEstimator(ABC):
    """Abstract base class representing a 6D Pose Estimation method."""

    @abstractmethod
    def estimate_pose(
        self,
        pcd: o3d.geometry.PointCloud,
        cad_mesh: o3d.geometry.TriangleMesh,
        **kwargs
    ) -> np.ndarray | None:
        """
        Estimates the 6D pose of the CAD model relative to the point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): Reconstructed scene point cloud in robot frame.
            cad_mesh (o3d.geometry.TriangleMesh): Reference CAD model. # AGENT: is this the most agnostic class we can think of ? I feel that some 0 shot method won't even need the CAD mesh file
            **kwargs: Method-specific inputs (e.g., rgb_crop, depth_crop, camera intrinsics).

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix, or None if estimation fails.
        """
        pass
