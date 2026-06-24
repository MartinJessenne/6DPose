from PIL.ImageOps import crop
from huggingface_hub import HfApi, hf_hub_download, login
from datasets import load_dataset, Dataset, load_from_disk
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch 
import cv2
import io
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

    bbox =result.boxes.xyxy[0].round().int() # Extract the rounded integer coordinates of the bounding box [Num_Instances, 4]
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

    return blacked_out_rgb_cropped, blacked_out_cropped_depth

def extract_depth_mask(depth_tensor, mask) -> torch.Tensor:
    """
    this function takes a torch.tensor (depth_array) and a torch.tensor (mask) e.g. produced by yolo_mask
    and extracts the corresponding depth patch to the mask as torch.tensor
    """
    return depth_tensor[mask[0]]
    # ok so here we've learned something important, 
    # since mask is torch.size([1280, 800]) but the 1 values inside it are a chaotic shape, when you try to do mask indexing
    # the result is not a [1280, 800] shape, otherwise what values would be the masked out coordinates? 
    # the best solution, to me is the following
    # for best efficiency first, crop the image to the cart bounding box size
    # extract the cart segmentation mask
    # crop the depth map to the bounding box value
    # extract the cart segmentation mask from the cropped depth map
    # this allows two things :
    # 1. more efficiency by dealing with lower resolution tensor (a fraction of the original (1280, 800) shape) 
    # 2. to still be able to manipulate a regular depth image and use classic algorithms on it 

# todo: note that there is a big limitation which is that currently the model is not robust to input with more than 1 cart

def instance_detected(result):
    """
    This function takes as input an ultralytics.engine.results.Results
    and output a boolean if there is a segmented cart instance in the result
    """
    if result[0].masks is not None:
        return True
    else:
        return False

if __name__ == "__main__":
    model = load_hf_model()

    local_dataset = load_from_disk("./train")

    img= local_dataset["rgb"][0]
    depth_bytes = local_dataset["depth"][0]
    depth_1d = np.frombuffer(depth_bytes, np.float32)
    depth_tensor = torch.tensor(depth_1d.reshape((1280, 800)).copy())

    # run the inference on the sample
    result = model(img, retina_masks=True)

    if instance_detected(result):
        cropped_rgb, cropped_depth = yolo_mask(result, depth_tensor)
        numpy_depth_mask = cropped_depth.numpy()
        numpy_cropped_rgb = cropped_rgb.numpy()
        plt.figure(figsize=(10,16))
        plt.imshow(numpy_cropped_rgb)
        plt.show()



    # print(local_dataset)
    # 
    # # ok great so now, we have to extract the segment mask from the input picture with the model
    # # and apply this mask to the depth map
    # mask = yolo_mask(input_rgb)
