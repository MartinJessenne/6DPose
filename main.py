import os
from huggingface_hub import HfApi, hf_hub_download, login
from datasets import load_dataset, Dataset, load_from_disk
from ultralytics import YOLO
import numpy as np
import torch 
import open3d as o3d
import cv2
import trimesh

def load_parquet_dataset():

    dataset_stream = load_dataset(
        "parquet",
        data_files={
            "train": "dataset/data/train-*-of-00127.parquet",
            "validation": "dataset/data/validation-*-of-00016.parquet",
            "test": "dataset/data/test-*-of-00016.parquet"
        },
        streaming=True,
        )
        
    head = Dataset.from_list(list(dataset_stream["test"].take(100)))

    return head


def load_hf_model():
    local_path = "best.pt"
    if not os.path.exists(local_path):
        print("Downloading model from Hugging Face...")
        cached_path = hf_hub_download("UItraviolet/yolo_multicart", "runs/segment/train-2/weights/best.pt")
        import shutil
        shutil.copy(cached_path, local_path)
        print(f"Model saved locally to {local_path}")
    else:
        print(f"Loading model from local path: {local_path}")

    model = YOLO(local_path)

    print("finished loading the model")
    return model 

def compute_bbox_area(bbox: torch.Tensor):
    """
    This function takes as input a torch.Tensor with shape [x, y, w, h]
    and outputs the value of the area of that box in (pixel unit)
    """
    return bbox[-1] * bbox[-2]

def select_target_detection(result):
    """
    Selects the detection with the largest bounding box area from the YOLO result.
    Returns:
        class_name (str): The recognized class name.
        bbox (list): [xmin, ymin, xmax, ymax] coordinates.
        mask (torch.Tensor): The segmentation mask of shape [H, W] on CPU.
    """
    result = result[0]
    
    # Select the bounding box with the largest area
    idx, _ = max(enumerate(result.boxes.xywh), key=lambda pair: compute_bbox_area(pair[1]))
    
    # Get the class name
    class_id = int(result.boxes.cls[idx].item())
    class_name = result.names[class_id]
    
    # Get coordinates
    bbox = result.boxes.xyxy[idx].round().int().tolist()
    
    # Get pixel mask
    mask = result.masks[idx].data.bool().squeeze(0).cpu()
    
    return class_name, bbox, mask


def crop_and_mask_inputs(orig_img, mask, depth_tensor, bbox):
    """
    Crops the original image, depth tensor, and mask, and zeroes out the background pixels.
    Returns:
        blacked_out_rgb_cropped (torch.Tensor)
        blacked_out_cropped_depth (torch.Tensor)
        xmin (int)
        ymin (int)
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Convert orig_img to torch.Tensor if it is a numpy array
    if not isinstance(orig_img, torch.Tensor):
        orig_img = torch.tensor(orig_img, dtype=torch.uint8)
        
    rgb_cropped = orig_img[ymin:ymax, xmin:xmax, :]
    cropped_mask = mask[ymin:ymax, xmin:xmax]
    
    blacked_out_rgb_cropped = torch.where(cropped_mask.unsqueeze(-1), rgb_cropped, 0)
    
    cropped_depth = depth_tensor[ymin:ymax, xmin:xmax]
    blacked_out_cropped_depth = torch.where(cropped_mask, cropped_depth, 0)
    
    return blacked_out_rgb_cropped, blacked_out_cropped_depth, xmin, ymin

def instance_detected(result):
    """
    This function takes as input an ultralytics.engine.results.Results
    and output a boolean if there is a segmented cart instance in the result
    """
    if result[0].masks is not None:
        return True
    else:
        return False


class Context:
    def __init__(self, camera, xmin, ymin, width_orig, height_orig, width_crop, height_crop):
        self.camera = camera
        self.xmin = xmin
        self.ymin = ymin 
        self.width_orig = width_orig
        self.height_orig = height_orig
        self.width_crop = width_crop
        self.height_crop = height_crop

        self.crop_cx = self.camera.cx - self.xmin
        self.crop_cy = self.camera.cy - self.ymin

class Camera:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

def point_cloud_processing(rgb, depth, ctx):
    """
    This function takes as an input two numpy array of the same shape
    For now I'll try something naive, I'll just get the inverse matrix, for the rgb, and consider the associated pixel depth value as the Z value, 
    Return a point cloud in the camera's frame
    """

    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1., convert_rgb_to_intensity=False
    )

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=ctx.width_crop, height=ctx.height_crop,
        fx=ctx.camera.fx, fy=ctx.camera.fy, cx=ctx.crop_cx, cy=ctx.crop_cy
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    return pcd


def o3d_to_ppf_format(pcd):
    """Extracts position and normals into an Nx6 array required by PPF."""
    # If normals do not exist we estimate them
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
    
    pts = np.array(pcd.points, dtype=np.float32)
    normals= np.asarray(pcd.normals, dtype=np.float32)
    return np.hstack((pts, normals))

def combine_geometries(meshes, point_clouds):
    combined = o3d.geometry.TriangleMesh()
    for m in meshes:
        combined += m  # merges vertices/faces/colors, offsetting face indices automatically

    # Ensure combined mesh has vertex colors before appending point cloud colors
    if not combined.has_vertex_colors():
        combined.vertex_colors = o3d.utility.Vector3dVector(
            np.ones((len(combined.vertices), 3)) * 0.7  # default gray
        )

    verts = np.asarray(combined.vertices)
    colors = np.asarray(combined.vertex_colors)

    for pcd in point_clouds:
        pts = np.asarray(pcd.points)
        if pcd.has_colors():
            cols = np.asarray(pcd.colors)
        else:
            cols = np.ones((len(pts), 3)) * 0.5  # default gray if uncolored

        verts = np.vstack([verts, pts])
        colors = np.vstack([colors, cols])

    combined.vertices = o3d.utility.Vector3dVector(verts)
    combined.vertex_colors = o3d.utility.Vector3dVector(colors)
    return combined

def SixDPoseEstimation(pcd, cad_mesh):

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    pcd.orient_normals_towards_camera_location(camera_location=np.zeros(3))

    pcd.transform(T_robot_camera)

    cad_mesh.compute_vertex_normals()

    model_pc = cad_mesh.sample_points_uniformly(number_of_points=1_000)

    ppf_model = o3d_to_ppf_format(model_pc)
    ppf_scene = o3d_to_ppf_format(pcd)

    detector = cv2.ppf_match_3d_PPF3DDetector(relativeSamplingStep=0.05, relativeDistanceStep=0.05)

    print("Training PPF Hash Table from CAD model...")
    detector.trainModel(ppf_model)

    print("Running PPF Match on cropped D455 cloud...")

    result = detector.match(ppf_scene, 0.05, 0.03)

    best_match = result[0]
    T_ppf = best_match.pose
    score = best_match.numVotes

    print(f"PPF Alignment Complete. Best match votes: {score}")

    T_init = np.asarray(T_ppf, dtype=np.float64).reshape(4,4)

    threshold = 0.1
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)

    icp_result = o3d.pipelines.registration.registration_icp(
        model_pc,
        pcd,
        threshold,
        T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )

    T_final = icp_result.transformation
    return T_final



if __name__ == "__main__":
    """
    Global context infos :
    The input images have a landscape aspect ratio with a dimension of (H, W, C) = (800, 1280, 3) for the rgb input and (H, W) = (800, 1280) for the depht input !
    """

    model = load_hf_model()
    
    # Initiate the camera struct, a helper struct to access the intrinsic parameters of the camera
    camera = Camera(fx=639.99768, fy=639.99768, cx=640.0, cy=400.0)

    T_robot_camera = np.array([
        [0.5, 0., 0.866, 0.439],
        [0.0, 1.0, -0., 0.],
        [-0.866, 0., 0.5, 0.304],
        [0., 0., 0., 1.]
    ])

    local_dataset = load_parquet_dataset()

    # Overwrite/recreate the debug_output directory at the start of the script
    output_dir = "debug_output/"
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Select 10 random unique indices from the test set
    total_samples = len(local_dataset)
    num_samples_to_test = min(10, total_samples)
    random_indices = np.random.choice(total_samples, num_samples_to_test, replace=False)
    
    print(f"Starting test run on {num_samples_to_test} random test samples: {random_indices}")

    for loop_idx, random_sample_idx in enumerate(random_indices):
        print(f"\n--- Processing Sample {loop_idx + 1}/{num_samples_to_test} (Index {random_sample_idx}) ---")
        img = local_dataset["rgb"][random_sample_idx]

        depth_bytes = local_dataset["depth"][random_sample_idx]
        depth_1d = np.frombuffer(depth_bytes, np.float32)
        depth_tensor = torch.tensor(depth_1d.reshape((800, 1280)).copy())

        # run the inference on the sample
        result = model(img, retina_masks=True)

        if not instance_detected(result):
            print(f"Skipping Sample (Index {random_sample_idx}): No cart instance detected.")
            continue

        cart_type, bbox, mask = select_target_detection(result)
        
        ### IMPORTANT CAVEAT : I MESSED UP THE TRAINING AND INVERTED THE PICANOL AND COLRUYT LABELS, SO THEY ARE CHANGED HERE, THIS IS A MINOR CHANGE SINCE IT'S JUST A LABEL. 
        print(f"Recognized class: {cart_type}")

        cropped_rgb, cropped_depth, xmin, ymin = crop_and_mask_inputs(
            result[0].orig_img, mask, depth_tensor, bbox
        )
        numpy_depth_mask = cropped_depth.numpy()
        numpy_cropped_rgb = cropped_rgb.numpy()

        ctx = Context(camera=camera, xmin=xmin, ymin=ymin, width_orig=np.array(img).shape[1], height_orig=np.array(img).shape[0], width_crop=numpy_cropped_rgb.shape[1], height_crop=numpy_cropped_rgb.shape[0])

        pcd = point_cloud_processing(numpy_cropped_rgb, numpy_depth_mask, ctx)

        cad_mesh = o3d.io.read_triangle_mesh(f"meshes/{cart_type}.ply")

        T_final = SixDPoseEstimation(pcd, cad_mesh)

        if T_final is None:
            print(f"Skipping Sample (Index {random_sample_idx}): Pose estimation failed.")
            continue

        print("Final 6D Pose Matrix (Refined via ICP): \n", T_final)
        T_world_camera = np.asarray(local_dataset["camera_view_transform"][random_sample_idx]).reshape(4,4).T
        T_world_cart = np.asarray(local_dataset["bbox_3d_transform"][random_sample_idx][0]).reshape(4,4).T
        T_usd_to_cv = np.diag([1, -1, -1, 1])
        T_ground_truth = T_robot_camera @ T_usd_to_cv @ T_world_camera @ T_world_cart
        print("Ground Truth 6D Pose Matrix : \n", T_ground_truth)

        # =================================================================
        # VISUALIZATION & DEBUGGING BLOCK
        # =================================================================
        import copy

        # 1. Create a coordinate axis at the Robot Base Origin (0,0,0)
        # Red = X axis, Green = Y axis, Blue = Z axis
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # 2. Color your scene point cloud grey so it acts as a neutral background
        pcd_vis = copy.deepcopy(pcd)
        pcd_vis.paint_uniform_color([0.6, 0.6, 0.6])

        # 3. Create a point cloud copy for your Predicted Pose (Paint it GREEN)
        predicted_mesh = copy.deepcopy(cad_mesh)
        predicted_mesh.transform(T_final)
        predicted_mesh.paint_uniform_color([0.0, 1.0, 0.0]) # Solid Green

        # 4. Create a point cloud copy for Isaac Sim Ground Truth (Paint it BLUE)
        # We handle the Column-Major transposition explicitly here (.T)
        T_gt_robot = T_ground_truth
        
        gt_mesh = copy.deepcopy(cad_mesh)
        gt_mesh.transform(T_gt_robot)
        gt_mesh.paint_uniform_color([0.0, 0.0, 1.0]) # Solid Blue

        # 5. Save everything to disk
        scene = trimesh.Scene()
        for name, m in [("frame", world_frame), ("pred", predicted_mesh), ("gt", gt_mesh)]:
            tm = trimesh.Trimesh(
                vertices=np.asarray(m.vertices),
                faces=np.asarray(m.triangles),
                vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8)
                            if m.has_vertex_colors() else None,
            )
            scene.add_geometry(tm, geom_name=name)

        pts = np.asarray(pcd_vis.points)
        cols = (np.asarray(pcd_vis.colors) * 255).astype(np.uint8) if pcd_vis.has_colors() else None
        scene.add_geometry(trimesh.PointCloud(vertices=pts, colors=cols), geom_name="scene")

        output_file = f"{output_dir}/combined_scene_sample_{loop_idx}.glb"
        scene.export(output_file)

        print(f"Saved GLB for sample {loop_idx + 1} to {output_file}")

    print(f"\nAll operations completed. Results saved in the '{output_dir}' directory.")
