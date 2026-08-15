import argparse
import json
import os

import numpy as np
from PIL import Image

try:
    from sklearn.cluster import KMeans  # type: ignore[import]
except ImportError:
    print("Please install sklearn to use this script.")
    exit(1)

# Define the folder containing the image and JSON files
subfolder = "/path/to/your/dataset"
output_file = os.path.join(subfolder, "transforms.json")

# List to hold the frames
frames = []

# Iterate over the files in the folder
for file in sorted(os.listdir(subfolder)):
    if file.endswith(".json"):
        # Read the JSON file containing camera extrinsics and intrinsics
        json_path = os.path.join(subfolder, file)
        with open(json_path, "r") as f:
            data = json.load(f)

        # Read the corresponding image file
        image_file = file.replace(".json", ".png")
        image_path = os.path.join(subfolder, image_file)
        if not os.path.exists(image_path):
            print(f"Image file not found for {file}, skipping...")
            continue
        with Image.open(image_path) as img:
            w, h = img.size

        # Extract and normalize intrinsic matrix K
        K = data["K"]
        fx = K[0][0] * w
        fy = K[1][1] * h
        cx = K[0][2] * w
        cy = K[1][2] * h

        # Extract the transformation matrix
        transform_matrix = np.array(data["c2w"])
        # Adjust for OpenGL convention
        transform_matrix[..., [1, 2]] *= -1

        # Add the frame data
        frames.append(
            {
                "fl_x": fx,
                "fl_y": fy,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "file_path": f"./{os.path.relpath(image_path, subfolder)}",
                "transform_matrix": transform_matrix.tolist(),
            }
        )

# Create the output dictionary
transforms_data = {"orientation_override": "none", "frames": frames}

# Write to the transforms.json file
with open(output_file, "w") as f:
    json.dump(transforms_data, f, indent=4)

print(f"transforms.json generated at {output_file}")


# Train-test split function using K-means clustering with stride
def create_train_test_split(frames, n, output_path, stride):
    # Prepare the data for K-means
    positions = []
    for frame in frames:
        transform_matrix = np.array(frame["transform_matrix"])
        position = transform_matrix[:3, 3]  # 3D camera position
        direction = transform_matrix[:3, 2] / np.linalg.norm(
            transform_matrix[:3, 2]
        )  # Normalized 3D direction
        positions.append(np.concatenate([position, direction]))

    positions = np.array(positions)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(positions)
    centers = kmeans.cluster_centers_

    # Find the index closest to each cluster center
    train_ids = []
    for center in centers:
        distances = np.linalg.norm(positions - center, axis=1)
        train_ids.append(int(np.argmin(distances)))  # Convert to Python int

    # Remaining indices as test_ids, applying stride
    all_indices = set(range(len(frames)))
    remaining_indices = sorted(all_indices - set(train_ids))
    test_ids = [
        int(idx) for idx in remaining_indices[::stride]
    ]  # Convert to Python int

    # Create the split data
    split_data = {"train_ids": sorted(train_ids), "test_ids": test_ids}

    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"Train-test split file generated at {output_path}")


# Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate train-test split JSON file using K-means clustering."
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of frames to include in the training set.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for selecting test frames (not used with K-means).",
    )

    args = parser.parse_args()

    # Create train-test split
    train_test_split_path = os.path.join(subfolder, f"train_test_split_{args.n}.json")
    create_train_test_split(frames, args.n, train_test_split_path, args.stride)
