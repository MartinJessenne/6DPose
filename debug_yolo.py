import os
import cv2
from main import load_hf_model, load_parquet_dataset

def main():
    print("Loading model...")
    model = load_hf_model()
    
    print("Loading dataset...")
    dataset = load_parquet_dataset()
    
    output_dir = "debug_yolo_checks"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving debug images to '{output_dir}/'...\n")
    
    for idx in range(len(dataset)):
        # Retrieve the image (PIL Image format)
        img = dataset["rgb"][idx]
        
        # Run inference
        results = model(img, retina_masks=True)
        result = results[0]
        
        # Plot predictions (bounding boxes, masks, labels) onto the image
        annotated_img = result.plot()
        
        # Save the annotated image to disk
        out_path = os.path.join(output_dir, f"check_image_{idx}.png")
        cv2.imwrite(out_path, annotated_img)
        
        # Collect class names for console summary
        detected_classes = []
        if result.boxes is not None:
            for cls_id in result.boxes.cls.int().tolist():
                detected_classes.append(result.names[cls_id])
        
        print(f"-> Image {idx}: Saved to {out_path} | Detected: {detected_classes}")

    print(f"\nSanity check complete. You can inspect the images in the '{output_dir}/' folder.")

if __name__ == "__main__":
    main()
