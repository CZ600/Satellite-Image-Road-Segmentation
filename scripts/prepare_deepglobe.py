import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def crop_image(image, crop_size, overlap_ratio):
    rows, cols = image.shape[:2]
    row_tiles = int(np.ceil(rows / crop_size))
    col_tiles = int(np.ceil(cols / crop_size))
    row_tiles_between = row_tiles - (overlap_ratio - 1)
    col_tiles_between = col_tiles - (overlap_ratio - 1)
    if overlap_ratio != 1:
        row_tiles += row_tiles_between
        col_tiles += col_tiles_between

    row_stride = int((crop_size * row_tiles - rows) / (row_tiles - 1)) if row_tiles > 1 else 0
    col_stride = int((crop_size * col_tiles - cols) / (col_tiles - 1)) if col_tiles > 1 else 0
    row_step = crop_size - row_stride if row_tiles > 1 else crop_size
    col_step = crop_size - col_stride if col_tiles > 1 else crop_size

    tiles = []
    col_idx = 0
    for col in range(0, max(1, cols - crop_size + 2), col_step):
        col_idx += 1
        if col + crop_size > cols:
            col = max(0, cols - crop_size)
        row_idx = 0
        for row in range(0, max(1, rows - crop_size + 2), row_step):
            row_idx += 1
            if row + crop_size > rows:
                row = max(0, rows - crop_size)
            tile = image[row:row + crop_size, col:col + crop_size]
            if tile.shape[0] != crop_size or tile.shape[1] != crop_size:
                continue
            tiles.append((row_idx, col_idx, tile))
    return tiles


def process_split(source_data_dir, source_mask_dir, target_image_dir, target_mask_dir, crop_size, overlap_ratio):
    ensure_dir(target_image_dir)
    ensure_dir(target_mask_dir)

    image_files = sorted(source_data_dir.glob("*_sat.*"))
    if not image_files:
        raise RuntimeError(f"No DeepGlobe images found in {source_data_dir}")

    for image_path in tqdm(image_files, ncols=100, desc=f"Preparing {source_data_dir.parent.name}"):
        sample_id = image_path.stem.replace("_sat", "")
        mask_path = source_mask_dir / f"{sample_id}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read image/mask pair: {image_path}, {mask_path}")

        image_tiles = crop_image(image, crop_size, overlap_ratio)
        mask_tiles = crop_image(mask, crop_size, overlap_ratio)
        if len(image_tiles) != len(mask_tiles):
            raise RuntimeError(f"Tile count mismatch for {sample_id}")

        for (row_idx, col_idx, image_tile), (_, _, mask_tile) in zip(image_tiles, mask_tiles):
            image_name = f"{sample_id}_sat_{row_idx}_{col_idx}.png"
            mask_name = f"{sample_id}_mask_{row_idx}_{col_idx}.png"
            cv2.imwrite(str(target_image_dir / image_name), image_tile, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            cv2.imwrite(str(target_mask_dir / mask_name), mask_tile, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root folder of the external DeepGlobe split dataset.",
    )
    parser.add_argument(
        "--target-root",
        default="Datasets/DeepGlobe",
        help="Output folder under this repository.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="Crop size used to create dataset tiles.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    ensure_dir(target_root)

    split_specs = [
        ("train", "cropped_train", "cropped_train_labels", 2),
        ("val", "cropped_valid", "cropped_valid_labels", 1),
        ("test", "cropped_test", "cropped_test_labels", 1),
    ]

    for split_name, image_dir_name, mask_dir_name, overlap_ratio in split_specs:
        process_split(
            source_root / split_name / "data",
            source_root / split_name / "seg",
            target_root / image_dir_name,
            target_root / mask_dir_name,
            args.crop_size,
            overlap_ratio,
        )

    print(f"DeepGlobe dataset prepared at: {target_root.resolve()}")


if __name__ == "__main__":
    main()
