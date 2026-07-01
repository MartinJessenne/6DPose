from huggingface_hub import HfApi, hf_hub_download, login
from datasets import load_dataset, Dataset, load_from_disk
from ultralytics import YOLO
import numpy as np
import torch 
import open3d as o3d
import cv2

def load_hf_dataset():
    hf_repo = "uitraviolet/cart_dataset"

    login()
    api = HfApi()

    print(f"downloading first 10 samples from the {hf_repo} hf repo!")
    
    dataset_stream = load_dataset(hf_repo, split='train', streaming=True)

    head = dataset_stream.take(10)

    local_train = Dataset.from_generator(
        lambda: (yield from head),
        features=head.features
    )

    local_train.save_to_disk("./train")

    print("managed to save the 10 samples")


def load_hf_model():
    # first load the model from hugging face
    model_str = hf_hub_download("uitraviolet/yolo_cart_seg", "best.pt")

    model = YOLO(model_str)

    print("finished loading the model")
    return model 

def compute_bbox_area(bbox: torch.Tensor):
    """
    This function takes as input a torch.Tensor with shape [x, y, w, h]
    and outputs the value of the area of that box in (pixel unit)
    """
    return bbox[-1] * bbox[-2]

def yolo_mask(result, depth_tensor) -> [torch.Tensor] :
    """
    this function takes as input an ultralytics.engine.results.Results and a depth_tensor of shape [H, W]
    and outputs one torch.Tensor that corresponds to original image c torch.tensor (1, 1280, 800) of bools corresponding to a picture mask of the cart instance
    by applying the trained yolo segmentation model
    """

    result = result[0]
        
    # Get the initial rgb image : 
    orig_img = torch.tensor(result.orig_img, dtype=torch.uint8) # [H, W, C] 

    
    # here we make an OPINIONATED choice : we only keep the bbox with the largest area
    # The intuitive but risky choice is that it should correspond to the closest cart, which is the one we're interested in for docking
    # There are tons of scenario in which that might fail, e.g. one of the bbox flickers and gets downsized or upsized, this is a failing point to take into account

    idx, _ = max(enumerate(result.boxes.xywh), key=lambda pair: compute_bbox_area(pair[1]))

    bbox =result.boxes.xyxy[idx].round().int() # Extract the rounded integer coordinates of the bounding box [Num_Instances, 4]
    # Now output the cropped rgb and the segmentation mask
    xmin, ymin, xmax, ymax = bbox.tolist()
    rgb_cropped = orig_img[ymin:ymax, xmin:xmax, :]

    # Now extract the pixel mask 
    mask = result.masks[idx].data.bool().squeeze(0) # [H, W]

    mask = mask.cpu()

    # Crop the pixel mask
    cropped_mask = mask[ymin:ymax, xmin:xmax]

    # Black away the pixels of the cropped rgb not belonging to the pixel mask
    blacked_out_rgb_cropped = torch.where(cropped_mask.unsqueeze(-1), rgb_cropped, 0)

    # Crop the depth tensor 
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


if __name__ == "__main__":
    """
    Global context infos :
    The input images have a dimension of (H, W, C) = (1280, 800, 3) for the rgb input and (H, W) = (1200, 800) for the depht input !
    """

    model = load_hf_model()
    
    # Initiate the camera struct, a helper struct to access the intrinsic parameters of the camera
    camera = Camera(fx=639.99768, fy=639.99768, cx=400., cy=640.0)

    T_robot_camera = np.array([
        [0.5, 0., 0.866, 0.439],
        [0.0, 1.0, -0., 0.],
        [-0.866, 0., 0.5, 0.304],
        [0., 0., 0., 1.]
    ])

    local_dataset = load_from_disk("./train")

    img= local_dataset["rgb"][0]
    depth_bytes = local_dataset["depth"][0]
    depth_1d = np.frombuffer(depth_bytes, np.float32)
    depth_tensor = torch.tensor(depth_1d.reshape((1280, 800)).copy())

    # run the inference on the sample
    result = model(img, retina_masks=True)

    if instance_detected(result):
        cropped_rgb, cropped_depth, xmin, ymin = yolo_mask(result, depth_tensor)
        numpy_depth_mask = cropped_depth.numpy()
        numpy_cropped_rgb = cropped_rgb.numpy()

        ctx = Context(camera=camera, xmin=xmin, ymin=ymin, width_orig=np.array(img).shape[0], height_orig=np.array(img).shape[1], width_crop=numpy_cropped_rgb.shape[0], height_crop=numpy_cropped_rgb.shape[1])

        pcd = point_cloud_processing(numpy_cropped_rgb, numpy_depth_mask, ctx)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.zeros(3))

        pcd.transform(T_robot_camera)

        cad_mesh = o3d.io.read_triangle_mesh("meshes/picanol.ply")
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
        print("Final 6D Pose Matrix (Refined via ICP): \n", T_final)
        T_ground_truth = np.asarray(local_dataset["bbox_3d_transform"][0][0]).reshape(4,4).T
        print("Ground Truth 6D Pose Matrix : \n", T_ground_truth)

        # =================================================================
        # VISUALIZATION & DEBUGGING BLOCK
        # =================================================================
        import copy

        print("\n--- Launching 3D Debug Visualizer ---")
        print("Controls: Use your mouse to rotate/pan. Press 'N' to toggle surface normals.")
        
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
        T_gt_robot = np.asarray(local_dataset["bbox_3d_transform"][0][0]).reshape(4,4).T
        
        gt_mesh = copy.deepcopy(cad_mesh)
        gt_mesh.transform(T_gt_robot)
        gt_mesh.paint_uniform_color([0.0, 0.0, 1.0]) # Solid Blue

        # 5. Render everything together in a single interactive window
        o3d.visualization.draw_geometries(
            [world_frame, pcd_vis, predicted_mesh, gt_mesh],
            window_name="6D Pose Debugger: Green=Prediction, Blue=Ground Truth",
            width=1280,
            height=720
        )

