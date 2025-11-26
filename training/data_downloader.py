"""
Simple image-caption dataset downloader without HuggingFace datasets dependency.
Downloads COCO validation images and captions.
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from tqdm import tqdm


class COCODownloader:
    """Downloads COCO validation dataset for training."""

    def __init__(self, data_dir: str = "data/coco_val", max_samples: Optional[int] = None):
        """
        Initialize COCO downloader

        Args:
            data_dir: Directory to store downloaded data
            max_samples: Maximum number of samples to download (None = download all available)
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_file = self.data_dir / "annotations.json"
        self.max_samples = max_samples
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

    def download_annotations(self):
        """Download COCO validation annotations."""

        # COCO 2017 validation annotations
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        zip_path = self.data_dir / "annotations.zip"

        if not self.annotations_file.exists():
            print("Downloading annotations")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with open(zip_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Extract annotations
            print("Extracting annotations")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extract("annotations/captions_val2017.json", self.data_dir)
            (self.data_dir / "annotations" / "captions_val2017.json").rename(self.annotations_file)
            os.rmdir(self.data_dir / "annotations")
            zip_path.unlink()

        print("Annotations ready!")

    def download_images(self):
        """Download COCO validation images."""
        # Load annotations
        with open(self.annotations_file, "r") as f:
            data = json.load(f)

        images_info = {img["id"]: img for img in data["images"]}
        annotations = data["annotations"]

        image_captions = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in image_captions:
                image_captions[img_id] = []
            image_captions[img_id].append(ann["caption"])

        # Download images
        downloaded = 0
        base_url = "http://images.cocodataset.org/val2017/"

        for img_id, captions in tqdm(image_captions.items(), desc="Downloading images"):
            if self.max_samples is not None and downloaded >= self.max_samples:
                break

            img_info = images_info.get(img_id)
            if not img_info:
                continue

            filename = img_info["file_name"]
            img_path = self.images_dir / filename

            if img_path.exists():
                downloaded += 1
                continue

            # Download image
            try:
                img_url = base_url + filename
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    downloaded += 1
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                continue

        print(f"Downloaded {downloaded} images!")

    def prepare_dataset(self):
        """Prepare the complete dataset."""
        if not self.annotations_file.exists():
            self.download_annotations()

        # Download images
        self.download_images()

        print(f"\nDataset ready at {self.data_dir}")
        print(f"Images: {self.images_dir}")
        print(f"Annotations: {self.annotations_file}")


class SimpleCOCODataset:
    """Simple COCO dataset that loads from downloaded files."""

    def __init__(self, data_dir: str = "data/coco_val", max_samples: Optional[int] = None):
        """
        Initialize COCO dataset.

        Args:
            data_dir: Directory containing downloaded data
            max_samples: Maximum number of samples to load (None = load all available)
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_file = self.data_dir / "annotations.json"

        # Load annotations
        with open(self.annotations_file, "r") as f:
            data = json.load(f)
        self.samples = []
        images_info = {img["id"]: img for img in data["images"]}
        image_captions = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in image_captions:
                image_captions[img_id] = []
            image_captions[img_id].append(ann["caption"])

        # Create samples
        for img_id, captions in image_captions.items():
            if max_samples is not None and len(self.samples) >= max_samples:
                break
            img_info = images_info.get(img_id)
            if not img_info:
                continue
            img_path = self.images_dir / img_info["file_name"]
            if img_path.exists():
                self.samples.append({"image_path": str(img_path), "captions": captions})

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        caption = item["captions"][0]
        return {"image": image, "question": "Describe this image.", "answer": caption}


if __name__ == "__main__":
    downloader = COCODownloader(max_samples=None)
    downloader.prepare_dataset()

    dataset = SimpleCOCODataset(max_samples=None)
    print(f"\nDataset has {len(dataset)} samples")

    if len(dataset) > 0:
        sample = dataset[0]
        print("\nFirst sample:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Image size: {sample['image'].size}")
        print(f"Image size: {sample['image'].size}")
