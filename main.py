from sympy.plotting.series import flat
from matplotlib.pylab import axis
from PIL.ImageOps import crop
from huggingface_hub import HfApi, hf_hub_download, login
from datasets import load_dataset, Dataset, load_from_disk
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch 
import cv2
import io
import open3d as o3d
from PIL import Image


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


class context:
    def __init__(self, xmin, ymin, orig_H, orig_W):
        self.xmin = xmin
        self.ymin = ymin 
        self.orig_H = orig_H
        self.orig_W = orig_W

def point_cloud_processing(rgb, depth, ctx):
    """
    This function takes as an input two numpy array of the same shape
    For now I'll try something naive, I'll just get the inverse matrix, for the rgb, and consider the associated pixel depth value as the Z value, 
    I won't try to use the depth camera's instrinsic parameters' yet. 
    """

    rgb_width, rgb_height = rgb.shape[1], rgb.shape[0]
    rgb_fx, rgb_fy = 639.99768, 639.99768
    rgb_cx, rgb_cy = 640.0, 400.0


    # Here we're creating the coordinate matrix of the rgb input, this will allow us to back project it 
    # But in fact I'm really confused as to what I'm doing here, it feels like the rgb input is useless, we only need its shapes
    # Which I kind of understand since the rgb channel (the main info of the rgb input) is only useful if we want to make a colored point cloud...
    # Now I need to better understand how to remap the c_x and c_y variable to the new cropped image.
    # Saying that the (0,0) coordinates where the top-left corner of the original image, then the image center is located at (c_x, c_y)
    # Cropping the image results in a translation of the origin from (0, 0) to (xmin, ymin)
    # Thus, the translated camera center (c'_x, c'_y) verifies : 
    # c'_x = c_x - xmin
    # c'_y = c_y - ymin
    crop_cx = rgb_cx - ctx.xmin
    crop_cy = rgb_cy - ctx.ymin

    K_inv = np.array([
        [1/rgb_fx, 0, -crop_cx/rgb_fx],
        [0, 1/rgb_fy, -crop_cy/rgb_fy],
        [0, 0, 1]
    ])

    # To backproject everything, we'll need the indice matrix of the input images
    x_col_coord = np.arange(rgb.shape[1])
    y_row_coord = np.arange(rgb.shape[0])

    xx_coord, yy_coord = np.meshgrid(x_col_coord, y_row_coord)

    xx_coord = xx_coord[..., np.newaxis]
    yy_coord = yy_coord[..., np.newaxis]

    rgb_coordinates = np.concatenate((xx_coord, yy_coord, np.ones_like(xx_coord)), axis=-1)



    # We use the meshgrid function to create this with the correct indexing : 
    rgb_coordinates = rgb_coordinates[..., np.newaxis]

    # Now each coordinate is going to be point-wise multiplied by the corresponding depth and by K_inv
    back_proj = K_inv @ rgb_coordinates
    back_proj = back_proj.squeeze(-1)

    depth = depth[..., np.newaxis]
    point_cloud = depth * back_proj

    return point_cloud


if __name__ == "__main__":
    """
    Global context infos :
    The input images have a dimension of (H, W, C) = (1280, 800, 3) for the rgb input and (H, W) = (1200, 800) for the depht input !
    """

    model = load_hf_model()

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
        plt.figure(figsize=(10,16))
        plt.imshow(numpy_cropped_rgb)
        plt.show()

        ctx = context(xmin, ymin, np.array(img).shape[0], np.array(img).shape[1])

        point_cloud = point_cloud_processing(numpy_cropped_rgb, numpy_depth_mask, ctx)

        flat_cloud = point_cloud.reshape(-1, 3)
        valid_mask = (flat_cloud[:, 2] > 0) & (flat_cloud[:, 2] < 10) # The depth values are in meters
        flat_cloud = flat_cloud[valid_mask]
        flat_color = (numpy_cropped_rgb.reshape(-1, 3)/255.0)[valid_mask]

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(flat_cloud)
        pcd.colors= o3d.utility.Vector3dVector(flat_color)

        o3d.visualization.draw_geometries([pcd])
