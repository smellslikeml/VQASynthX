import torch
from diffusers import AutoPipelineForText2Image
from tqdm.auto import tqdm
from PIL import Image


def naive_parser(prompt: str, delimiter: str = "|"):
    """Splits prompt into content and style parts."""
    parts = prompt.split(delimiter)
    if len(parts) != 2:
        raise ValueError(
            "Prompt must contain exactly one '|' delimiter for content|style parsing."
        )
    return parts[0].strip(), parts[1].strip()


def generate_with_lpa_control(
    prompt: str,
    model_id: str = "stabilityai/sdxl-turbo",
    device: str = "cuda",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    style_control_end_step: float = 0.8,  # Apply style for the first 80% of steps
) -> Image.Image:
    """
    Generates an image using a simplified Local Prompt Adaptation (LPA) concept.
    This is a training-free method to control style and content by manipulating
    prompt embeddings during the denoising process. We use a full prompt (content + style)
    for early steps and a content-only prompt for later steps to refine the structure.

    Args:
        prompt (str): A prompt string with content and style separated by '|'.
                      Example: "a red cube next to a blue sphere | oil painting"
        model_id (str): The Hugging Face model ID for the diffusion pipeline.
        device (str): The device to run the model on ('cuda' or 'cpu').
        num_inference_steps (int): The number of denoising steps.
        guidance_scale (float): The guidance scale for classifier-free guidance.
        style_control_end_step (float): The fraction of steps during which to apply style.

    Returns:
        A PIL Image.
    """
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)

    content_prompt, style_prompt = naive_parser(prompt)
    full_prompt = f"{content_prompt}, {style_prompt}"

    # 1. Get prompt embeddings for both full and content-only prompts
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=full_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    content_only_embeds, _, content_only_pooled_embeds, _ = pipe.encode_prompt(
        prompt=content_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,  # Only need the conditional part
    )

    # We need to construct CFG-compatible embeddings for the content-only part
    content_cfg_embeds = torch.cat([negative_prompt_embeds, content_only_embeds])
    content_cfg_pooled_embeds = torch.cat(
        [negative_pooled_prompt_embeds, content_only_pooled_embeds]
    )

    # 2. Prepare latents
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor
    latents = pipe.prepare_latents(
        1, pipe.unet.config.in_channels, height, width, prompt_embeds.dtype, device
    )

    # 3. Denoising loop with LPA control
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    style_cutoff_step = int(num_inference_steps * style_control_end_step)

    add_time_ids = pipe._get_add_time_ids(
        (height, width),
        (0, 0),
        (height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    ).to(device)

    for i, t in enumerate(tqdm(timesteps)):
        if i < style_cutoff_step:
            current_prompt_embeds = prompt_embeds
            current_pooled_embeds = pooled_prompt_embeds
        else:
            current_prompt_embeds = content_cfg_embeds
            current_pooled_embeds = content_cfg_pooled_embeds

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        added_cond_kwargs = {
            "text_embeds": current_pooled_embeds,
            "time_ids": add_time_ids,
        }
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=current_prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # 4. Decode the latents
    image = pipe.vae.decode(
        latents / pipe.vae.config.scaling_factor, return_dict=False
    )[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return image
