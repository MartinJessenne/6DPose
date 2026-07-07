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
    OUTPUT_DIR = "debug_output/"


# =====================================================================
# 1b. METHOD-SPECIFIC HYPERPARAMETERS
# =====================================================================
class PPFICPParams:
    """Hyperparameters specific to the PPF + ICP matching method."""
    def __init__(
        self,
        ppf_sampling_step: float = 0.04,
        ppf_distance_step: float = 0.03,
        ppf_match_threshold: float = 0.03,
        ppf_match_tolerance: float = 0.01,
        icp_max_correspondence_distance: float = 0.09915124703931394,
        icp_max_iterations: int = 50
    ):
        """
        Initializes hyperparameters for PPF and ICP alignment.
        
        Args:
            ppf_sampling_step (float): Sampling step for training model and scene.
            ppf_distance_step (float): Distance step for training.
            ppf_match_threshold (float): Match threshold for PPF.
            ppf_match_tolerance (float): Match tolerance angle.
            icp_max_correspondence_distance (float): Max distance threshold for ICP correspondence.
            icp_max_iterations (int): Max number of iterations for ICP refinement.
            
        Example:
            >>> params = PPFICPParams(ppf_sampling_step=0.03, icp_max_iterations=100)
        """
        self.ppf_sampling_step = ppf_sampling_step
        self.ppf_distance_step = ppf_distance_step
        self.ppf_match_threshold = ppf_match_threshold
        self.ppf_match_tolerance = ppf_match_tolerance
        self.icp_max_correspondence_distance = icp_max_correspondence_distance
        self.icp_max_iterations = icp_max_iterations


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


def point_cloud_processing(rgb: np.ndarray, depth: np.ndarray, ctx: Context) -> o3d.geometry.PointCloud:
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
        color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
    )
    
    # Generate crop-adjusted camera intrinsics
    intrinsics = ctx.camera.get_o3d_intrinsics(
        width=ctx.width_crop,
        height=ctx.height_crop,
        xmin=ctx.xmin,
        ymin=ctx.ymin
    )
    
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)


def o3d_to_ppf_format(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Extracts 3D coordinates and surface normals from an Open3D point cloud
    into a stacked array format required by OpenCV's PPF matching detector.
    
    Args:
        pcd (o3d.geometry.PointCloud): Point cloud object.
        
    Returns:
        np.ndarray: Array of shape [N, 6] containing [x, y, z, nx, ny, nz].
    """
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
    
    pts = np.array(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return np.hstack((pts, normals))


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
    Performs 6D Pose estimation of a CAD model against a reconstructed scene point cloud.
    Steps include:
      1. Estimating normals and transforming the scene to the robot base frame.
      2. Uniformly sampling points from the CAD mesh.
      3. Training a PPF (Point Pair Features) detector and executing matching.
      4. Running ICP (Iterative Closest Point) refinement to compute the final 4x4 pose.
      
    Args:
        pcd (o3d.geometry.PointCloud): Reconstructed camera-frame point cloud.
        cad_mesh (o3d.geometry.TriangleMesh): The reference CAD model mesh.
        params (PPFICPParams, optional): Specific hyperparameters for PPF and ICP.
        
    Returns:
        np.ndarray: The 4x4 homogeneous transformation matrix aligning CAD model to robot base.
        
    Example:
        >>> params = PPFICPParams(icp_max_iterations=100)
        >>> T_final = SixDPoseEstimation(pcd, picanol_mesh, params=params)
    """
    if params is None:
        params = PPFICPParams()

    # Prepare scene point cloud (estimate normals & transform to robot base frame)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.zeros(3))
    pcd.transform(Config.T_ROBOT_CAMERA)
    
    # Prepare CAD model
    cad_mesh.compute_vertex_normals()
    model_pc = cad_mesh.sample_points_uniformly(number_of_points=1000)
    
    # Convert geometries to PPF stacked formats [N, 6]
    ppf_model = o3d_to_ppf_format(model_pc)
    ppf_scene = o3d_to_ppf_format(pcd)
    
    # Initialize PPF 3D Detector (Notice the legacy underscore convention used by cv2)
    detector = cv2.ppf_match_3d_PPF3DDetector(
        relativeSamplingStep=params.ppf_sampling_step,
        relativeDistanceStep=params.ppf_distance_step
    )
    
    print("Training PPF Hash Table from CAD model...")
    detector.trainModel(ppf_model)
    
    print("Running PPF Match on scene...")
    match_results = detector.match(
        ppf_scene,
        params.ppf_match_threshold,
        params.ppf_match_tolerance
    )
    
    # Retrieve the best match result
    best_match = match_results[0]
    T_init = np.asarray(best_match.pose, dtype=np.float64).reshape(4, 4)
    print(f"PPF Alignment Complete. Best match votes: {best_match.numVotes}")
    
    # ICP Refinement (Point-to-Plane registration)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=params.icp_max_iterations
    )
    icp_result = o3d.pipelines.registration.registration_icp(
        model_pc,
        pcd,
        params.icp_max_correspondence_distance,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    
    return icp_result.transformation


# =====================================================================
# 5b. HIGH-LEVEL PROCESS AND EXPORT UTILITIES
# =====================================================================
def process_and_reconstruct(
    img: np.ndarray,
    depth_bytes: bytes,
    result,
    camera: Camera
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
    pcd = point_cloud_processing(numpy_cropped_rgb, numpy_depth_mask, ctx)
    
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
    # Initialize model, camera, and dataset configurations
    model = load_hf_model()
    camera = Camera(
        fx=Config.CAMERA_FX, fy=Config.CAMERA_FY,
        cx=Config.CAMERA_CX, cy=Config.CAMERA_CY
    )
    local_dataset = load_parquet_dataset()
    method_params = PPFICPParams()
    
    # Recreate the output directory to clear old runs
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting test run on the test set : {len(local_dataset["test"])} test samples")
    
    for sample_idx, sample in enumerate(local_dataset["test"]):
        
        # Load sample raw data
        img = local_dataset["rgb"][sample_idx]
        depth_bytes = local_dataset["depth"][sample_idx]
        
        # Run YOLO inference
        result = model(img, retina_masks=True)
        if not instance_detected(result):
            print(f"Skipping Sample (Index {sample_idx}): No cart instance detected.")
            continue
            
        # 1. Segment and Reconstruct Point Cloud
        cart_type, pcd = process_and_reconstruct(img, depth_bytes, result, camera)
        print(f"Recognized class: {cart_type}")
        
        # Load reference CAD mesh
        mesh_file = f"meshes/{cart_type}.ply"
        if not os.path.exists(mesh_file):
            print(f"Skipping Sample (Index {sample_idx}): CAD file {mesh_file} not found.")
            continue
        cad_mesh = o3d.io.read_triangle_mesh(mesh_file)
        
        # 2. Run 6D Pose Estimation
        T_final = SixDPoseEstimation(pcd, cad_mesh, params=method_params)
        if T_final is None:
            print(f"Skipping Sample (Index {sample_idx}): Pose estimation failed.")
            continue
        print("Final 6D Pose Matrix (Refined via ICP):\n", T_final)
        
        # 3. Calculate Ground Truth pose
        T_ground_truth = compute_ground_truth_pose(local_dataset, sample_idx)
        print("Ground Truth 6D Pose Matrix:\n", T_ground_truth)
        
        # 4. Assemble and Export Debugging Scene to Disk
        output_file = os.path.join(Config.OUTPUT_DIR, f"combined_scene_sample_{sample_idx}.glb")
        export_debug_scene(pcd, cad_mesh, T_final, T_ground_truth, output_file)
        
    print(f"\nAll operations completed. Results saved in the '{Config.OUTPUT_DIR}' directory.")
