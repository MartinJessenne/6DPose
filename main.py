import os
import shutil
import copy
import numpy as np
import torch
import open3d as o3d
import cv2
import trimesh
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# =====================================================================
# 1. CENTRALIZED PIPELINE CONFIGURATION
# =====================================================================
class Config:
    """Centralized parameters, paths, and transformation matrices for the pipeline."""
    # Hugging Face Repository Settings
    HF_REPO = "UItraviolet/yolo_multicart"
    HF_FILE = "runs/segment/train-2/weights/best.pt"
    LOCAL_MODEL_PATH = "best.pt"
    
    # Camera Intrinsics Parameters
    CAMERA_FX = 639.99768
    CAMERA_FY = 639.99768
    CAMERA_CX = 640.0
    CAMERA_CY = 400.0
    
    # Extrinsic Camera-to-Robot base_link Transform
    T_ROBOT_CAMERA = np.array([
        [0.5, 0.0,  0.866, 0.439],
        [0.0, 1.0, -0.0,   0.0  ],
        [-0.866, 0.0, 0.5, 0.304],
        [0.0, 0.0,  0.0,   1.0  ]
    ])
    
    # Dataset and Output Settings
    DATASET_PATH = "parquet"
    TRAIN_PARQUET_GLOB = "dataset/data/train-*-of-00127.parquet"
    VAL_PARQUET_GLOB = "dataset/data/validation-*-of-00016.parquet"
    TEST_PARQUET_GLOB = "dataset/data/test-*-of-00016.parquet"
    NUM_SAMPLES_TO_LOAD = 100
    NUM_SAMPLES_TO_TEST = 10
    DEFAULT_DEPTH_TRUNC = 3.2
    OUTPUT_DIR = "debug_output/"



# =====================================================================
# 1b. METHOD-SPECIFIC HYPERPARAMETERS
# =====================================================================
# Imported from methods for backward compatibility
from methods import PPFICPParams



# =====================================================================
# 2. DATASET AND MODEL LIFECYCLE HELPERS
# =====================================================================
def load_parquet_dataset() -> Dataset:
    """
    Loads the parquet dataset splits, streams them, and returns a local slice of the test set.
    
    Returns:
        Dataset: A Hugging Face Dataset containing sample rows from the test set.
        
    Example:
        >>> dataset = load_parquet_dataset()
        >>> first_sample = dataset[0]
        >>> print(first_sample.keys())
    """
    dataset_stream = load_dataset(
        Config.DATASET_PATH,
        data_files={
            "train": Config.TRAIN_PARQUET_GLOB,
            "validation": Config.VAL_PARQUET_GLOB,
            "test": Config.TEST_PARQUET_GLOB
        },
        streaming=True,
    )
    
    # Extract the test split and materialize the first N samples
    samples = list(dataset_stream["test"].take(Config.NUM_SAMPLES_TO_LOAD))
    return Dataset.from_list(samples)


def load_hf_model() -> YOLO:
    """
    Loads the YOLO segmentation model from a local file. If the file is not present,
    it downloads it from the Hugging Face Hub, saves it locally, and then loads it.
    
    Returns:
        YOLO: The initialized Ultralytics YOLO segmentation model.
        
    Example:
        >>> model = load_hf_model()
        >>> result = model("test_image.png")
    """
    local_path = Config.LOCAL_MODEL_PATH
    if not os.path.exists(local_path):
        print(f"Model not found locally at {local_path}. Downloading from Hugging Face...")
        cached_path = hf_hub_download(Config.HF_REPO, Config.HF_FILE)
        import shutil
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


def select_target_detection(result) -> tuple[str, list[int], torch.Tensor]:
    """
    Selects the YOLO prediction with the largest bounding box area from the result.
    
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


def crop_and_mask_inputs(
    orig_img: np.ndarray | torch.Tensor,
    mask: torch.Tensor,
    depth_tensor: torch.Tensor,
    bbox: list[int]
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Crops the original RGB image, depth tensor, and segmentation mask using the bounding box,
    and masks out any background pixels (pixels not covered by the segmentation mask).
    
    Args:
        orig_img: The original RGB image array.
        mask (torch.Tensor): Binary segmentation mask of shape [H, W].
        depth_tensor (torch.Tensor): Raw depth tensor of shape [H, W].
        bbox (list[int]): Coordinates [xmin, ymin, xmax, ymax].
        
    Returns:
        tuple containing:
            - blacked_out_rgb_cropped (torch.Tensor): The cropped, masked RGB image.
            - blacked_out_cropped_depth (torch.Tensor): The cropped, masked depth tensor.
            - xmin (int): Bounding box horizontal crop offset.
            - ymin (int): Bounding box vertical crop offset.
            
    Example:
        >>> cropped_rgb, cropped_depth, xmin, ymin = crop_and_mask_inputs(img, mask, depth_tensor, bbox)
    """
    xmin, ymin, xmax, ymax = bbox
    
    if not isinstance(orig_img, torch.Tensor):
        orig_img = torch.tensor(orig_img, dtype=torch.uint8)
        
    # Crop RGB image and segmentation mask
    rgb_cropped = orig_img[ymin:ymax, xmin:xmax, :]
    cropped_mask = mask[ymin:ymax, xmin:xmax]
    
    # Black out background pixels
    blacked_out_rgb_cropped = torch.where(cropped_mask.unsqueeze(-1), rgb_cropped, 0)
    
    # Crop and black out depth values
    cropped_depth = depth_tensor[ymin:ymax, xmin:xmax]
    blacked_out_cropped_depth = torch.where(cropped_mask, cropped_depth, 0)
    
    return blacked_out_rgb_cropped, blacked_out_cropped_depth, xmin, ymin


def instance_detected(result) -> bool:
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

    def get_o3d_intrinsics(
        self, width: int, height: int, xmin: int = 0, ymin: int = 0
    ) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Generates Open3D camera intrinsics, shifting the principal point to account
        for cropping offsets.
        
        Args:
            width (int): Width of the cropped image.
            height (int): Height of the cropped image.
            xmin (int): Horizontal crop offset.
            ymin (int): Vertical crop offset.
            
        Returns:
            o3d.camera.PinholeCameraIntrinsic: Shipped camera intrinsics.
            
        Example:
            >>> camera = Camera(640.0, 640.0, 320.0, 240.0)
            >>> intrinsics = camera.get_o3d_intrinsics(100, 100, 50, 50)
        """
        crop_cx = self.cx - xmin
        crop_cy = self.cy - ymin
        return o3d.camera.PinholeCameraIntrinsic(
            width=width, height=height,
            fx=self.fx, fy=self.fy, cx=crop_cx, cy=crop_cy
        )


class Context:
    """Holds geometric context information about the original and cropped views."""
    def __init__(
        self, camera: Camera, xmin: int, ymin: int,
        width_orig: int, height_orig: int,
        width_crop: int, height_crop: int
    ):
        self.camera = camera
        self.xmin = xmin
        self.ymin = ymin
        self.width_orig = width_orig
        self.height_orig = height_orig
        self.width_crop = width_crop
        self.height_crop = height_crop

        self.crop_cx = self.camera.cx - self.xmin
        self.crop_cy = self.camera.cy - self.ymin


def point_cloud_processing(
    rgb: np.ndarray,
    depth: np.ndarray,
    ctx: Context,
    depth_trunc: float = Config.DEFAULT_DEPTH_TRUNC
) -> o3d.geometry.PointCloud:

    """
    Converts RGB and depth arrays into an Open3D PointCloud object using the cropped intrinsics.
    
    Args:
        rgb (np.ndarray): Cropped and masked RGB image array.
        depth (np.ndarray): Cropped and masked depth image array.
        ctx (Context): The geometric cropping context.
        
    Returns:
        o3d.geometry.PointCloud: Reconstructed point cloud in the camera's local coordinate frame.
    """
    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, 
        depth_o3d, 
        depth_scale=1.0,
        depth_trunc=depth_trunc, 
        convert_rgb_to_intensity=False,
    )


    
    # Generate crop-adjusted camera intrinsics
    intrinsics = ctx.camera.get_o3d_intrinsics(
        width=ctx.width_crop,
        height=ctx.height_crop,
        xmin=ctx.xmin,
        ymin=ctx.ymin
    )
    
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)





# =====================================================================
# 5. COORDINATE TRANSFORMS AND 6D POSE ESTIMATION
# =====================================================================
def compute_ground_truth_pose(local_dataset: Dataset, sample_idx: int) -> np.ndarray:
    """
    Retrieves the ground truth cart pose, maps it from the Isaac Sim arbitrary world frame 
    to the camera coordinate frame (adjusting for USD to OpenCV conventions), and projects 
    it into the robot's base_link frame.
    
    Args:
        local_dataset (Dataset): The materialized local dataset.
        sample_idx (int): The index of the sample.
        
    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix in the robot's base frame.
        
    Example:
        >>> T_gt = compute_ground_truth_pose(local_dataset, 5)
    """
    # 1. Load raw flat lists and reshape into 4x4 row-major matrices
    T_world_camera = np.asarray(local_dataset["camera_view_transform"][sample_idx]).reshape(4, 4).T
    T_world_cart = np.asarray(local_dataset["bbox_3d_transform"][sample_idx][0]).reshape(4, 4).T
    
    # 2. Define USD (Z-back, Y-up) to OpenCV (Z-forward, Y-down) coordinate change matrix
    T_usd_to_cv = np.diag([1, -1, -1, 1])
    
    # 3. Compute final coordinate multiplication chain
    return Config.T_ROBOT_CAMERA @ T_usd_to_cv @ T_world_camera @ T_world_cart


def SixDPoseEstimation(
    pcd: o3d.geometry.PointCloud,
    cad_mesh: o3d.geometry.TriangleMesh,
    params: PPFICPParams = None
) -> np.ndarray:
    """
    Legacy wrapper around the modular PPFICPEstimator for backward compatibility.
    """
    from methods import PPFICPEstimator
    estimator = PPFICPEstimator(params=params)
    return estimator.estimate_pose(pcd, cad_mesh)



# =====================================================================
# 5b. HIGH-LEVEL PROCESS AND EXPORT UTILITIES
# =====================================================================
def process_and_reconstruct(
    img: np.ndarray,
    depth_bytes: bytes,
    result,
    camera: Camera,
    depth_trunc: float = Config.DEFAULT_DEPTH_TRUNC
) -> tuple[str, o3d.geometry.PointCloud]:

    """
    Extracts the target cart segmentation mask, crops depth and RGB data,
    and reconstructs the 3D point cloud of the target instance in the camera frame.
    
    Args:
        img (np.ndarray): Original raw RGB image.
        depth_bytes (bytes): The raw depth buffer bytes.
        result: The YOLO Results object.
        camera (Camera): The Camera intrinsics instance.
        
    Returns:
        tuple containing:
            - cart_type (str): The recognized class name.
            - pcd (o3d.geometry.PointCloud): Reconstructed 3D point cloud in camera frame.
            
    Example:
        >>> cart_type, pcd = process_and_reconstruct(img, depth_bytes, result, camera)
    """
    # Prepare depth tensor
    depth_1d = np.frombuffer(depth_bytes, np.float32)
    depth_tensor = torch.tensor(depth_1d.reshape((800, 1280)).copy())
    
    # Select target mask and crop parameters
    cart_type, bbox, mask = select_target_detection(result)
    
    # Crop and mask inputs
    cropped_rgb, cropped_depth, xmin, ymin = crop_and_mask_inputs(
        result[0].orig_img, mask, depth_tensor, bbox
    )
    numpy_depth_mask = cropped_depth.numpy()
    numpy_cropped_rgb = cropped_rgb.numpy()
    
    # Reconstruct 3D Point Cloud using camera properties and context
    ctx = Context(
        camera=camera, xmin=xmin, ymin=ymin,
        width_orig=np.array(img).shape[1], height_orig=np.array(img).shape[0],
        width_crop=numpy_cropped_rgb.shape[1], height_crop=numpy_cropped_rgb.shape[0]
    )
    pcd = point_cloud_processing(numpy_cropped_rgb, numpy_depth_mask, ctx, depth_trunc=depth_trunc)

    
    return cart_type, pcd


def export_debug_scene(
    pcd: o3d.geometry.PointCloud,
    cad_mesh: o3d.geometry.TriangleMesh,
    T_final: np.ndarray,
    T_ground_truth: np.ndarray,
    output_path: str
) -> None:
    """
    Constructs a debugging 3D scene containing the reconstructed point cloud, the 
    predicted pose mesh (green), the ground truth pose mesh (blue), and the origin 
    coordinate frame. Exports the combined geometries as a GLB file.
    
    Args:
        pcd (o3d.geometry.PointCloud): Reconstructed scene point cloud (in robot base frame).
        cad_mesh (o3d.geometry.TriangleMesh): Reference CAD mesh.
        T_final (np.ndarray): Final 4x4 predicted transformation matrix.
        T_ground_truth (np.ndarray): Ground truth 4x4 transformation matrix.
        output_path (str): File path to save the GLB export.
        
    Example:
        >>> export_debug_scene(pcd, cad_mesh, T_final, T_gt, "debug_output/scene.glb")
    """
    # 1. Create origin coordinate axes (Red=X, Green=Y, Blue=Z)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # 2. Create neutral background color for point cloud
    pcd_vis = copy.deepcopy(pcd)
    pcd_vis.paint_uniform_color([0.6, 0.6, 0.6])
    
    # 3. Transform CAD mesh for predicted pose (GREEN)
    predicted_mesh = copy.deepcopy(cad_mesh)
    predicted_mesh.transform(T_final)
    predicted_mesh.paint_uniform_color([0.0, 1.0, 0.0])
    
    # 4. Transform CAD mesh for ground truth pose (BLUE)
    gt_mesh = copy.deepcopy(cad_mesh)
    gt_mesh.transform(T_ground_truth)
    gt_mesh.paint_uniform_color([0.0, 0.0, 1.0])
    
    # 5. Build trimesh scene and export
    scene = trimesh.Scene()
    for name, m in [("frame", world_frame), ("pred", predicted_mesh), ("gt", gt_mesh)]:
        tm = trimesh.Trimesh(
            vertices=np.asarray(m.vertices),
            faces=np.asarray(m.triangles),
            vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8)
                          if m.has_vertex_colors() else None
        )
        scene.add_geometry(tm, geom_name=name)
        
    pts = np.asarray(pcd_vis.points)
    cols = (np.asarray(pcd_vis.colors) * 255).astype(np.uint8) if pcd_vis.has_colors() else None
    scene.add_geometry(trimesh.PointCloud(vertices=pts, colors=cols), geom_name="scene")
    
    scene.export(output_path)
    print(f"Saved GLB scene to {output_path}")


# =====================================================================
# 6. ORCHESTRATION PIPELINE (MAIN EXECUTION BLOCK)
# =====================================================================
if __name__ == "__main__":
    print("main.py is now a utility module. To run the pose estimation and inspection scripts, please run inspect_pose.py or benchmark.py instead.")


