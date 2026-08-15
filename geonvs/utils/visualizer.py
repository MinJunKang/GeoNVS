
import cv2
import torch
from einops import rearrange
import numpy as np
from PIL import Image, ImageDraw
import torch.nn.functional as F
from sklearn.decomposition import PCA


def pca_latents(latents: torch.Tensor, size_img) -> np.ndarray:
    pca = PCA(n_components=3)
    h_, w_ = latents.shape[-2], latents.shape[-1]
    latents_pca = rearrange(latents, "b c h w -> (b h w) c").cpu().numpy()
    pca.fit(latents_pca)
    pca_features = pca.transform(latents_pca)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = rearrange(pca_features, "(b h w) c -> b c h w", h=h_, w=w_)
    frames = torch.tensor(pca_features).to(latents.device)
    frames = F.interpolate(frames, size=size_img, mode='bilinear')
    feat_pca = np.uint8(frames.permute(0, 2, 3, 1).float().cpu().numpy() * 255)
    video_frames = [feat for feat in feat_pca]
    return video_frames


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path,
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)