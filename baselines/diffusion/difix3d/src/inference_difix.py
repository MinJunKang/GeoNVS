import os
import imageio
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from model import Difix


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to the reference image or directory')
    parser.add_argument('--height', type=int, default=576, help='Height of the input image')
    parser.add_argument('--width', type=int, default=1024, help='Width of the input image')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to be used')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument('--video', action='store_true', help='If the input is a video')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=args.model_path,
        timestep=args.timestep,
        mv_unet=True if args.ref_image is not None else False,
    )
    model.set_eval()

    # Load input images
    if os.path.isdir(args.input_image):
        input_images = sorted(glob(os.path.join(args.input_image, "*.png")))
    else:
        input_images = [args.input_image]

    # Load reference images if provided
    if args.ref_image is not None:
        if os.path.isdir(args.ref_image):
            ref_images = sorted(glob(os.path.join(args.ref_image, "*")))
        else:
            ref_images = [args.ref_image]

        assert len(input_images) == len(ref_images), "Number of input images and reference images should be the same"

    # Process images
    output_images = []
    for i, input_image in enumerate(tqdm(input_images, desc="Processing images")):
        image = Image.open(input_image).convert('RGB')
        ref_image = Image.open(ref_images[i]).convert('RGB') if args.ref_image is not None else None
        output_image = model.sample(
            image,
            height=args.height,
            width=args.width,
            ref_image=ref_image,
            prompt=args.prompt
        )
        output_images.append(output_image)

    # Save outputs
    if args.video:
        # Save as video
        video_path = os.path.join(args.output_dir, "output.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        for output_image in tqdm(output_images, desc="Saving video"):
            writer.append_data(np.array(output_image))
        writer.close()
    else:
        # Save as individual images
        for i, output_image in enumerate(tqdm(output_images, desc="Saving images")):
            output_image.save(os.path.join(args.output_dir, os.path.basename(input_images[i])))