from __future__ import annotations
import os
import shutil
import copy
import numpy as np
import torch
import open3d as o3d
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# =====================================================================
# SYSTEM ARCHITECTURE AND DATAFLOW DIAGRAM
# =====================================================================
#
# MODULE INTERACTION & DEPENDENCY GRAPH
# ------------------------------------
#
#       [ inspect_pose.py ]               [ benchmark.py ]
#        (CLI Inspection)                 (CLI Benchmarking)
#                |                                |
#                +---------------+----------------+
#                                |
#                                v
#                           [ pipeline.py ] <---+
#                  (Pipeline Utilities)    |
#                                |          | (Imported for
#                                v          |  T_ROBOT_CAMERA)
#                          [ methods/ ] ----+
#                     (Modular Estimators)
#                               |
#         +---------------------+---------------------+
#         |                     |                     |
#         v                     v                     v
#     [ base.py ]        [ ppf.py ]           [ ransac.py ]
#   (Base Estimator)    (PPF + Dual ICP)      (RANSAC + Dual ICP)
#
#
# PIPELINE DATAFLOW
# -----------------
#
#        +-------------------------------------------+
#        |           Parquet Test Dataset            |
#        +---------------------+---------------------+
#                              |
#              +---------------+---------------+
#              |                               |
#              v (PIL Image)                   v (Binary Depth Bytes)
#      [ rgb_image ]                    [ depth_bytes ]
#              |                               |
#              v                               v
#   +----------------------+               +----------------------+
#   |  YOLO Segmentation   |               |   Reshape & Convert  |
#   |  (best.pt model)     |               |   to float32 Tensor  |
#   +----------+-----------+               +-----------+----------+
#              |                                       |
#              | [class, bbox, mask]                   | [depth_tensor]
#              +-------------------+   +---------------+
#                                  |   |
#                                  v   v
#                      +---------------------------+
#                      |   crop_and_mask_inputs    |
#                      | - Crop image/depth        |
#                      | - Keep only mask region   |
#                      +-------------+-------------+
#                                    |
#                                    v [cropped & masked rgb/depth]
#                      +---------------------------+
#                      |  point_cloud_processing   |
#                      | - Shift camera principal  |
#                      |   point by crop offset    |
#                      | - Open3D PCD generation   |
#                      +-------------+-------------+
#                                    |
#                                    v [PCD in camera frame]
#                      +---------------------------+
#                      |    BasePoseEstimator      |
#                      |      (estimate_pose)      |
#                      +-------------+-------------+
#                                    |
#          +-------------------------+-------------------------+
#          |                                                   |
#          v                                                   v
#   +---------------------------------+                 +---------------------------------+
#   |          PPFEstimator           |                 |         RansacEstimator         |
#   |                                 |                 |                                 |
#   | 1. Normal Estimation            |                 | 1. Normal Estimation            |
#   | 2. Transform to Robot Frame     |                 | 2. Transform to Robot Frame     |
#   |    (T_ROBOT_CAMERA)             |                 |    (T_ROBOT_CAMERA)             |
#   | 3. Uniform CAD Mesh Sampling    |                 | 3. Uniform CAD Mesh Sampling    |
#   | 4. Train opencv_ppf hash table  |                 | 4. Voxel Downsample both PCDs   |
#   | 5. cv2 PPF 3D Match on Scene    |                 | 5. Compute FPFH descriptors     |
#   | 6. Dual ICP Alignment:          |                 | 6. RANSAC Feature Matching      |
#   |    - Hyp 1: Raw PPF pose        |                 | 7. Dual ICP Alignment:          |
#   |    - Hyp 2: 180° rotation along |                 |    - Hyp 1: Raw RANSAC pose     |
#   |             local Z-axis        |                 |    - Hyp 2: 180° rotation along |
#   | 7. Evaluate Overlap & Fitness   |                 |             local Z-axis        |
#   | 8. Select Best Pose (T_final)   |                 | 8. Evaluate Overlap & Fitness   |
#   +---------------------------------+                 | 9. Select Best Pose (T_final)   |
#                                     |                 +---------------------------------+
#                                     |                                 |
#                                     +----------------+----------------+
#                                                      |
#                                                      v [T_final (Predicted Pose)]
#                                                      |
#                      +-------------------------------+---------------+
#                      |                                               |
#                      |                   +---------------------------+
#                      |                   |
#                      |                   v
#                      |       +---------------------------------------+
#                      |       |       compute_ground_truth_pose       |
#                      |       | - Extract USD-based matrix from data  |
#                      |       | - Transform USD coordinates to OpenCV |
#                      |       | - Transform OpenCV to Robot Frame     |
#                      |       +-------------------+-------------------+
#                      |                           |
#                      |                           v [T_ground_truth]
#                      |                           |
#                      v                           v
#        +-------------------------------------------+
#        |            export_debug_scene             |
#        | - Re-color Point Cloud to Gray            |
#        | - Transform CAD Mesh to Pred Pose (Green) |
#        | - Transform CAD Mesh to GT Pose (Blue)    |
#        | - Write composite 3D Scene to GLB File    |
#        +-------------------------------------------+
#

# =====================================================================
# 1. CENTRALIZED PIPELINE CONFIGURATION
# =====================================================================
# Configuration is managed via tyro (see cli_config.py).



# 2. DATASET AND MODEL LIFECYCLE HELPERS
# =====================================================================
def load_parquet_dataset(dataset_path: str = None, test_glob: str = None) -> Dataset:
    """
    Loads the parquet test dataset split directly in Map-style format (non-streaming).
    
    Returns:
        Dataset: The loaded Hugging Face test split.
        
    Example:
        >>> dataset = load_parquet_dataset()
        >>> first_sample = dataset[0]
        >>> print(first_sample.keys())
    """
    path = dataset_path if dataset_path is not None else "parquet"
    glob_pattern = test_glob if test_glob is not None else "dataset/data/test-*-of-00016.parquet"
    dataset = load_dataset(
        path,
        data_files={
            "test": glob_pattern
        },
        streaming=False,
    )
    return dataset["test"]


def load_hf_model(local_model_path: str = None, repo_id: str = None, filename: str = None) -> YOLO:
    """
    Loads the YOLO segmentation model from a local file. If the file is not present,
    it downloads it from the Hugging Face Hub, saves it locally, and then loads it.
    
    Returns:
        YOLO: The initialized Ultralytics YOLO segmentation model.
        
    Example:
        >>> model = load_hf_model()
        >>> result = model("test_image.png")
    """
    local_path = local_model_path if local_model_path is not None else "best.pt"
    repo = repo_id if repo_id is not None else "UItraviolet/yolo_multicart"
    file = filename if filename is not None else "runs/segment/train-2/weights/best.pt"
    # Check if the YOLO model exists locally. If not, download it from Hugging Face
    # and copy it to the local project path.
    if not os.path.exists(local_path):
        print(f"Model not found locally at {local_path}. Downloading from Hugging Face...")
        cached_path = hf_hub_download(repo, file)
        # shutil is a standard Python library used here to copy the cached model file to the local directory
        shutil.copy(cached_path, local_path) 
        print(f"Model saved locally to {local_path}")
    else:
        print(f"Loading model from local path: {local_path}")
        
    model = YOLO(local_path)
    print("finished loading the model")
    return model


# =====================================================================
# 3. YOLO POST-PROCESSING & SEGMENTATION HELPERS
# =====================================================================
def compute_bbox_area(bbox: torch.Tensor) -> float:
    """
    Computes the area of a bounding box.
    
    Args:
        bbox (torch.Tensor): A tensor representing coordinates [x, y, w, h].
        
    Returns:
        float: The area of the bounding box in pixels.
        
    Example:
        >>> bbox = torch.tensor([10.0, 20.0, 50.0, 80.0])
        >>> area = compute_bbox_area(bbox)
        >>> print(area)  # Output: 4000.0
    """
    return float(bbox[-1] * bbox[-2])


def select_target_detection(result: list) -> tuple[str, list[int], torch.Tensor]:
    """
    Selects the YOLO prediction with the largest bounding box area from the result.
    This is the initial opinionated choice we make. Later on, we might compose that with a lookup_cart variable, e.g. if the model detects a picanol and a colruyt cart, while lookup_cart = picanol, 
    It will only recognize and output the detected picanol and ignore all other carts class.
    And if there are 2 picanols it will fallback to the current mechanism, retaining the one with the largest bounding box area, 
    even there need to consider the edge case where the two bounding boxes have the exact same area. 
    
    Args:
        result: The YOLO Results object for a single image.
        
    Returns:
        tuple containing:
            - class_name (str): The name of the predicted class (e.g. 'picanol').
            - bbox (list[int]): Bounding box coordinates [xmin, ymin, xmax, ymax].
            - mask (torch.Tensor): The segmentation mask of shape [H, W] on CPU.
            
    Example:
        >>> results = model(img)
        >>> class_name, bbox, mask = select_target_detection(results)
    """
    # Assumption: The input `result` list contains detection results for a single image.
    # For a real-time/streaming multi-image batch pipeline, this indexing must be updated
    # to iterate over the batch elements robustly.
    result_img = result[0]
    
    # Enumerate the bounding boxes and select the index with the largest area
    idx, _ = max(
        enumerate(result_img.boxes.xywh),
        key=lambda pair: compute_bbox_area(pair[1])
    )
    
    # Retrieve the class ID and look up its string name
    class_id = int(result_img.boxes.cls[idx].item())
    class_name = result_img.names[class_id]
    
    # Retrieve coordinates
    bbox = result_img.boxes.xyxy[idx].round().int().tolist()
    
    # Extract the binary mask
    mask = result_img.masks[idx].data.bool().squeeze(0).cpu()
    
    return class_name, bbox, mask


class MaskedImageFrame:
    """
    Encapsulates cropped and masked sensor observations alongside their matching camera model.

    Design Choice Rationale:
    ------------------------
    This class is introduced to enforce coordinate alignment safety at the type/class level (a pattern
    common in systems languages like Rust).
    In the previous implementation, cropped image buffers and camera principal point offsets were handled
    independently. If point cloud reconstruction was called elsewhere, there was no structural guarantee
    that the camera intrinsics were shifted by the correct crop offsets matching the image shape.
    By packaging the cropped RGB-D data, the original camera, and the bounding box offsets into a single
    immutable-like frame, we guarantee that the correct crop-adjusted camera intrinsics are always computed
    dynamically in get_o3d_intrinsics().
    """
    def __init__(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        camera: Camera,
        xmin: int,
        ymin: int
    ):
        self.rgb = rgb
        self.depth = depth
        self.camera = camera
        self.xmin = xmin
        self.ymin = ymin

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    def get_o3d_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Generates Open3D camera intrinsics, shifting the principal point to account
        for cropping offsets.
        """
        crop_cx = self.camera.cx - self.xmin
        crop_cy = self.camera.cy - self.ymin
        return o3d.camera.PinholeCameraIntrinsic(
            width=self.width, height=self.height,
            fx=self.camera.fx, fy=self.camera.fy,
            cx=crop_cx, cy=crop_cy
        )


def crop_and_mask_inputs(
    orig_img: torch.Tensor,
    mask: torch.Tensor,
    depth_tensor: torch.Tensor,
    bbox: list[int],
    camera: Camera
) -> MaskedImageFrame:
    """
    Crops the original RGB image, depth tensor, and segmentation mask using the bounding box,
    and masks out any background pixels. Returns a packaged MaskedImageFrame.

    Args:
        orig_img (torch.Tensor): Original RGB image tensor of shape [H, W, 3].
        mask (torch.Tensor): Binary segmentation mask of shape [H, W].
        depth_tensor (torch.Tensor): Raw depth tensor of shape [H, W].
        bbox (list[int]): Bounding box coordinates [xmin, ymin, xmax, ymax].
        camera (Camera): Original camera model for intrinsics calculation.

    Returns:
        MaskedImageFrame: Packaged crop and mask results.
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Crop RGB image and segmentation mask
    rgb_cropped = orig_img[ymin:ymax, xmin:xmax, :]
    cropped_mask = mask[ymin:ymax, xmin:xmax]
    
    # Black out background pixels
    blacked_out_rgb_cropped = torch.where(cropped_mask.unsqueeze(-1), rgb_cropped, 0)
    
    # Crop and black out depth values
    cropped_depth = depth_tensor[ymin:ymax, xmin:xmax]
    blacked_out_cropped_depth = torch.where(cropped_mask, cropped_depth, 0)
    
    return MaskedImageFrame(
        rgb=blacked_out_rgb_cropped,
        depth=blacked_out_cropped_depth,
        camera=camera,
        xmin=xmin,
        ymin=ymin
    )


def instance_detected(result: list) -> bool:
    """
    Checks if any segmented instances were detected by the YOLO model in the inference result.
    
    Args:
        result: The YOLO Results object list.
        
    Returns:
        bool: True if at least one instance has a valid segmentation mask, False otherwise.
    """
    return result[0].masks is not None


# =====================================================================
# 4. CAMERA INTRINSICS AND POINT CLOUD GENERATION CLASSES
# =====================================================================
class Camera:
    """Represents a pinhole camera model and manages coordinates for image cropping."""
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


def point_cloud_processing(
    frame: MaskedImageFrame,
    depth_trunc: float = 3.0
) -> o3d.geometry.PointCloud:
    """
    Converts a MaskedImageFrame into an Open3D PointCloud object using the cropped intrinsics.
    
    Args:
        frame (MaskedImageFrame): Packaged cropped and masked image frame.
        depth_trunc (float): Max depth to include in the point cloud.
        
    Returns:
        o3d.geometry.PointCloud: Reconstructed point cloud in the cameras local coordinate frame.
    """
    # Convert PyTorch tensors back to numpy for Open3D integration
    rgb_np = frame.rgb.numpy()
    depth_np = frame.depth.numpy()

    color_o3d = o3d.geometry.Image(rgb_np)
    depth_o3d = o3d.geometry.Image(depth_np)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, 
        depth_o3d, 
        depth_scale=1.0,
        depth_trunc=depth_trunc, 
        convert_rgb_to_intensity=False,
    )
    
    intrinsics = frame.get_o3d_intrinsics()
    
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)





# =====================================================================
# 5. COORDINATE TRANSFORMS AND 6D POSE ESTIMATION
# =====================================================================
def compute_ground_truth_pose(T_world_camera: np.ndarray, T_world_cart: np.ndarray, T_robot_camera: np.ndarray) -> np.ndarray:
    """
    Computes the ground truth 6D pose of the cart in the robot's base frame.
    
    The function transforms the cart pose from global world coordinates (arbitrary USD origin)
    to the camera coordinate frame (adjusting for USD to OpenCV conventions), and projects 
    it into the robot's base_link frame.
    
    Args:
        T_world_camera (np.ndarray): 4x4 transform from world to camera.
        T_world_cart (np.ndarray): 4x4 transform from world to cart.
        T_robot_camera (np.ndarray): 4x4 extrinsic transform from camera to robot base link.
        
    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix in the robot's base frame.
    """
    # 2. Define USD (Z-back, Y-up) to OpenCV (Z-forward, Y-down) coordinate change matrix
    T_usd_to_cv = np.diag([1, -1, -1, 1])
    
    # 3. Compute final coordinate multiplication chain
    return T_robot_camera @ T_usd_to_cv @ T_world_camera @ T_world_cart


# =====================================================================
# 5b. HIGH-LEVEL PROCESS AND EXPORT UTILITIES
# =====================================================================
def process_and_reconstruct(
    img: np.ndarray,
    depth_bytes: bytes,
    result: list,
    camera: Camera,
    depth_trunc: float = 3.0
) -> tuple[str, o3d.geometry.PointCloud]:
    """
    Extracts the target cart segmentation mask, crops depth and RGB data,
    and reconstructs the 3D point cloud of the target instance in the camera frame.
    
    Args:
        img (np.ndarray): Original raw RGB image.
        depth_bytes (bytes): The raw depth buffer bytes.
        result: The YOLO Results object.
        camera (Camera): The Camera intrinsics instance.
        depth_trunc (float): Max depth to include in the point cloud.
        
    Returns:
        tuple containing:
            - cart_type (str): The recognized class name (e.g., 'picanol').
            - pcd (o3d.geometry.PointCloud): Reconstructed 3D point cloud in camera frame.
    """
    # Prepare depth tensor dynamically using the shape of the original image
    depth_1d = np.frombuffer(depth_bytes, np.float32)
    img_np = np.array(img)
    height, width = img_np.shape[:2]
    depth_tensor = torch.tensor(depth_1d.reshape((height, width)).copy())
    
    # Select target mask and crop parameters
    cart_type, bbox, mask = select_target_detection(result)
    
    # Convert original RGB image (from BGR YOLO result) to a PyTorch tensor on CPU
    orig_img_tensor = torch.from_numpy(result[0].orig_img[..., ::-1].copy())
    
    # Crop and mask inputs, producing the type-safe MaskedImageFrame
    frame = crop_and_mask_inputs(
        orig_img=orig_img_tensor,
        mask=mask,
        depth_tensor=depth_tensor,
        bbox=bbox,
        camera=camera
    )
    
    # Reconstruct 3D Point Cloud using camera properties and MaskedImageFrame
    pcd = point_cloud_processing(frame, depth_trunc=depth_trunc)
    
    return cart_type, pcd


def load_cad_meshes(meshes_dir: str = None) -> dict[str, o3d.geometry.TriangleMesh]:
    """
    Loads all PLY CAD meshes from the meshes/ directory.
    
    Args:
        meshes_dir: Absolute path to the meshes directory. If None, uses 
                    the 'meshes/' directory relative to this script.
    
    Returns:
        dict mapping cart type name (e.g. 'colruyt') to the loaded TriangleMesh.
        
    Raises:
        RuntimeError: If no meshes are found.
    """
    import glob as _glob
    if meshes_dir is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        meshes_dir = os.path.join(project_root, "meshes")
    
    mesh_pattern = os.path.join(meshes_dir, "*.ply")
    mesh_files = _glob.glob(mesh_pattern)
    meshes = {}
    for f in mesh_files:
        name = os.path.splitext(os.path.basename(f))[0]
        meshes[name] = o3d.io.read_triangle_mesh(f)
    if not meshes:
        raise RuntimeError(
            f"No meshes found matching pattern: {mesh_pattern}. "
            f"Ensure the meshes/ directory is populated."
        )
    return meshes
