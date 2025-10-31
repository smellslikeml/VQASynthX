import torch
from . import model
from . import utils
from . import solver
from . import prune


def apply_patch(
    pipe,
    sx,
    sy,
    max_downsample,
    acc_range,
    lagrange_int,
    lagrange_step,
    lagrange_term,
    max_fix,
    max_interval,
    latent_size=None,
):
    """Applies the SADA patch to a diffusers pipeline."""
    if not hasattr(pipe, "unet"):
        raise ValueError("The provided pipe does not have a 'unet' attribute.")

    unet = pipe.unet

    # Create the cache bus for communication between modules
    cache_bus = utils.Cache_Bus(
        acc_range=acc_range,
        max_downsample=max_downsample,
        sx=sx,
        sy=sy,
        latent_size=latent_size,
        lagrange_int=lagrange_int,
        lagrange_step=lagrange_step,
        lagrange_term=lagrange_term,
        max_fix=max_fix,
        max_interval=max_interval,
    )

    # Wrap the UNet with our custom model
    unet_wrapper = model.SADAUNet(unet, cache_bus)
    pipe.unet = unet_wrapper

    # Store the cache bus on the model for later access (e.g., resetting)
    unet._cache_bus = cache_bus

    print("SADA patch applied successfully.")


def reset_cache(pipe):
    if hasattr(pipe, "unet") and hasattr(pipe.unet, "_cache_bus"):
        pipe.unet._cache_bus.reset()
        print("SADA cache reset.")
    else:
        print("Could not find SADA cache to reset.")
