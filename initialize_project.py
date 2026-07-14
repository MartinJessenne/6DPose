import os
import shutil
from huggingface_hub import login, hf_hub_download, snapshot_download

def main():
    print("====================================================")
    print("6D Pose Project Initialization & Asset Downloader")
    print("====================================================")
    
    # 1. Hugging Face Authentication Check
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info.get('username', 'Unknown')}")
    except Exception:
        print("You are not currently logged in to Hugging Face.")
        token = input("Please enter your Hugging Face Access Token: ").strip()
        if token:
            login(token=token)
        else:
            print("No token provided. Proceeding with public download access...")

    # 2. Download YOLO Segmentation Model
    model_repo = "UItraviolet/yolo_multicart"
    model_file = "runs/segment/train-2/weights/best.pt"
    local_model_path = "best.pt"
    
    if os.path.exists(local_model_path):
        print(f"\n[YOLO Model] Found locally at '{local_model_path}'. Skipping download.")
    else:
        print(f"\n[YOLO Model] Downloading '{model_file}' from '{model_repo}'...")
        try:
            cached_path = hf_hub_download(repo_id=model_repo, filename=model_file)
            shutil.copy(cached_path, local_model_path)
            print(f"[YOLO Model] Successfully saved to '{local_model_path}'")
        except Exception as e:
            print(f"[YOLO Model] Error downloading model: {e}")

    # 3. Download Parquet Dataset
    dataset_repo = "UItraviolet/industrial_cart"
    local_dataset_dir = "dataset/data"
    
    # Check if we already have parquet files locally
    has_parquet = False
    if os.path.exists(local_dataset_dir):
        has_parquet = any(
            f.endswith('.parquet')
            for root, _, files in os.walk(local_dataset_dir)
            for f in files
        )

    if has_parquet:
        print(f"\n[Dataset] Parquet files already exist in '{local_dataset_dir}'. Skipping download.")
    else:
        print(f"\n[Dataset] Downloading parquet dataset from '{dataset_repo}' (dataset type)...")
        os.makedirs(local_dataset_dir, exist_ok=True)
        try:
            # Download the parquet files using snapshot_download
            snapshot_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                local_dir=local_dataset_dir,
                allow_patterns="*.parquet"
            )
            print(f"[Dataset] Successfully downloaded dataset parquet files to '{local_dataset_dir}'")
        except Exception as e:
            print(f"[Dataset] Error downloading dataset: {e}")
            print("Please ensure your Hugging Face token has access to the private dataset repository.")

    print("\nInitialization finished. You can now run the project using:")
    print("  uv run inspect_pose.py mode=random random_samples=5 model=ransac")

if __name__ == "__main__":
    main()
