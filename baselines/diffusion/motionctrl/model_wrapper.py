
from omegaconf import OmegaConf
from baselines.diffusion.motionctrl.sgm.util import instantiate_from_config


def load_model(
    config: str,
    ckpt: str,
    device: str,
    num_frames: int,
    num_steps: int,
):

    config = OmegaConf.load(config)
    config.model.params.ckpt_path = ckpt
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )

    model = instantiate_from_config(config.model)

    model = model.to(device).eval()    

    filter = None #DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter