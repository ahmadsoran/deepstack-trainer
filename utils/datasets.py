import os
import glob
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ExifTags
from tqdm import tqdm

# Logging
logger = logging.getLogger(__name__)

# Acceptable formats
IMG_FORMATS = {'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff'}
VID_FORMATS = {'mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv'}

# Orientation exif tag
EXIF_ORIENTATION = {v: k for k, v in ExifTags.TAGS.items()}.get("Orientation", None)


def exif_size(img: Image.Image):
    """Return exif-corrected PIL size (w, h)."""
    s = img.size
    try:
        if hasattr(img, "_getexif") and img._getexif():
            rotation = img._getexif().get(EXIF_ORIENTATION, None)
            if rotation in [6, 8]:  # 90/270 degrees
                s = (s[1], s[0])
    except Exception:
        pass
    return s


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints.

    Returns:
        im (ndarray): padded image
        ratio (float): width/height scaling ratio
        (dw, dh) (tuple): padding added (width, height)
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    shape = im.shape[:2]  # current shape [height, width]
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # Resize
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def img2label_paths(img_paths):
    """Derive label paths from image paths (for YOLO txt labels)."""
    return [p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep, 1).rsplit(".", 1)[0] + ".txt"
            for p in img_paths]


def yolov5_collate_fn(batch):
    """Collate function compatible with YOLO-style datasets.

    Returns a batch of (imgs, labels, paths) where:
      - imgs: Tensor of shape (B, C, H, W)
      - labels: concatenated labels tensor of shape (N, 5)
      - paths: list of image paths
    """
    imgs = torch.stack([x[0] for x in batch])
    labels_list = [x[1] for x in batch]
    if len(labels_list) > 0:
        try:
            labels = torch.cat(labels_list, 0)
        except Exception:
            labels = torch.zeros((0, 5), dtype=torch.float32)
    else:
        labels = torch.zeros((0, 5), dtype=torch.float32)
    paths = [x[2] for x in batch]
    return imgs, labels, paths


class SimpleImageDataset(Dataset):
    """Minimal YOLO-style dataset loader (works on macOS)."""

    def __init__(self, path, img_size=640, augment=False):
        super().__init__()
        self.img_size = img_size
        self.augment = augment

        # Collect files
        files = []
        path = Path(path)
        if path.is_dir():
            files = glob.glob(str(path / "**" / "*.*"), recursive=True)
        elif path.is_file():
            with open(path) as f:
                files = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Dataset path {path} does not exist")

        # Filter image files
        self.img_files = [f for f in files if f.split(".")[-1].lower() in IMG_FORMATS]
        self.label_files = img2label_paths(self.img_files)
        assert self.img_files, f"No images found in {path}"

        # Load labels and image shapes for compatibility with training utils
        self.labels = []  # list of numpy arrays, one per image, shape (n_labels,5)
        shapes = []
        for img_path, lbl_path in zip(self.img_files, self.label_files):
            # Image shape
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
            h0, w0 = img.shape[:2]
            shapes.append([h0, w0])

            # Labels
            if os.path.exists(lbl_path):
                try:
                    lb = np.loadtxt(lbl_path, dtype=np.float32)
                    if lb.size == 0:
                        lb = np.zeros((0, 5), dtype=np.float32)
                    elif lb.ndim == 1:
                        lb = lb.reshape(1, -1)
                except Exception:
                    # fallback: read line by line
                    with open(lbl_path) as f:
                        lines = [list(map(float, line.split())) for line in f if line.strip()]
                    if len(lines) == 0:
                        lb = np.zeros((0, 5), dtype=np.float32)
                    else:
                        lb = np.array(lines, dtype=np.float32)
            else:
                lb = np.zeros((0, 5), dtype=np.float32)

            self.labels.append(lb)

        self.shapes = np.array(shapes, dtype=np.float64)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, i):
        # Load image
        img_path = self.img_files[i]
        img = cv2.imread(img_path)
        assert img is not None, f"Image not found: {img_path}"
        h0, w0 = img.shape[:2]

        # Resize
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

        # Convert to tensor
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img)

        # Load labels if available
        labels = []
        if os.path.exists(self.label_files[i]):
            with open(self.label_files[i]) as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.split())
                    labels.append([cls, x, y, w, h])
        labels = torch.tensor(labels, dtype=torch.float32)

        return img_tensor, labels, img_path


def create_dataloader(path, imgsz=640, batch_size=16, gs=32, opt=None, hyp=None,
                      augment=False, cache=False, rect=False, rank=-1,
                      world_size=1, workers=8, image_weights=False, pad=0.5):
    """Create a DataLoader compatible with the training script.

    This is a simplified implementation that wraps SimpleImageDataset while
    accepting the same parameters used by train.py. Some advanced features
    (rectangular batching, caching, image-weights) are not implemented here
    but the function signature is compatible.
    """
    dataset = SimpleImageDataset(path, img_size=imgsz, augment=augment)

    # Shuffle when not using rectangular training and not in distributed mode
    shuffle = not rect and rank in (-1, 0)

    # Use top-level collate function to ensure picklability with multiprocessing
    from utils.datasets import yolov5_collate_fn

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=min(workers, os.cpu_count() or 1),
                        pin_memory=True,
                        collate_fn=yolov5_collate_fn)

    return loader, dataset
