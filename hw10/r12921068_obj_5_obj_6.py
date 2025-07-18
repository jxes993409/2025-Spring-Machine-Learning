import itertools
import math
import os
import random
from pathlib import Path

import accelerate
import numpy as np
import safetensors
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from safetensors.torch import load_file

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import CustomDiffusionAttnProcessor, CustomDiffusionAttnProcessor2_0
from diffusers.optimization import get_scheduler

class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
    ):
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            inst_img_path = [
                (x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = list(zip(class_images_path, class_prompt))
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

def collate_fn(examples):
    """
    Puts together a batch of data.

    Args:
        examples (list of dict): A list of dictionaries, where each dictionary
            contains the keys "instance_prompt_ids", "instance_images", and "mask".

    Returns:
        dict: A dictionary containing the batched "input_ids", "pixel_values", and "mask".
              "input_ids" is a concatenated tensor of prompt token IDs.
              "pixel_values" is a stacked tensor of instance images.
              "mask" is a stacked tensor of masks with an added channel dimension.
    """
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    input_ids = torch.cat(input_ids, dim=0) # Concatenate prompt token IDs along the batch dimension.
    pixel_values = torch.stack(pixel_values) # Stack individual image tensors to form the image batch.
    mask = torch.stack(mask) # Stack individual mask tensors to form the mask batch.
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float() # Ensure data is in contiguous memory for potentially faster processing and convert to float.
    mask = mask.to(memory_format=torch.contiguous_format).float() # Ensure mask is in contiguous memory and convert to float.

    batch = {"input_ids": input_ids, "pixel_values": pixel_values, "mask": mask.unsqueeze(1)} # Add a channel dimension to the mask to be compatible with image processing layers.
    return batch

def save_new_embed(text_encoder, modifier_token_id, accelerator, modifier_token, output_dir, safe_serialization=True):
    """Saves the new token embeddings learned by the text encoder.

    Args:
        text_encoder: The trained text encoder model.
        modifier_token_id (list of int): List of token IDs corresponding to the modifier tokens.
        accelerator: The accelerator object used for distributed training.
        modifier_token (list of str): List of the modifier tokens (e.g., ["<new1>"]).
        output_dir (str): The directory where the new embeddings will be saved.
        safe_serialization (bool, optional): Whether to use safe serialization (safetensors). Defaults to True.
    """
    # Unwrap the potentially distributed text encoder model to access its components.
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight

    # Iterate through the modifier tokens and their corresponding IDs.
    for x, y in zip(modifier_token_id, [modifier_token]):
        # Create a dictionary to store the learned embedding for the current modifier token.
        learned_embeds_dict = {}
        # Extract the learned embedding for the specific modifier token ID.
        learned_embeds_dict[y] = learned_embeds[x]

        # Determine the filename and saving method based on the safe_serialization flag.
        if safe_serialization:
            filename = f"{output_dir}/{y}.safetensors"
            # Save the learned embedding dictionary using the safetensors format.
            safetensors.torch.save_file(learned_embeds_dict, filename, metadata={"format": "pt"})
        else:
            filename = f"{output_dir}/{y}.bin"
            # Save the learned embedding dictionary using the standard PyTorch format.
            torch.save(learned_embeds_dict, filename)

def train_func(
    output_dir,
    instance_prompt,
    instance_data_dir,
    freeze_model,
    learning_rate,
    max_train_steps,
    train_batch_size
):
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=Path(output_dir, "logs"))

    # create accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("custom-diffusion")

    # Handle the repository creation
    if accelerator.is_main_process and (output_dir is not None):
        os.makedirs(output_dir, exist_ok=True)

    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None, variant=None
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=None, variant=None
    )

    # Adding a modifier token which is optimized ####
    modifier_token_id = []
    initializer_token_id = []
    modifier_token = "<new1>"
    initializer_token = "ktn"

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(modifier_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {modifier_token}. Please pass a different"
            " `modifier_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)

    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id.append(token_ids[0])
    modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for x, y in zip(modifier_token_id, initializer_token_id):
        token_embeds[x] = token_embeds[y]

    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    vae.requires_grad_(False)
    if modifier_token is None:
        text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16" and modifier_token is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    attention_class = (
        CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CustomDiffusionAttnProcessor
    )

    # now we will add new Custom Diffusion weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
    train_kv = True
    train_q_out = False if freeze_model == "crossattn_kv" else True
    custom_diffusion_attn_procs = {}

    st = unet.state_dict()
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
            "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
        }
        if train_q_out:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=train_kv,
                train_q_out=train_q_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=False,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
    del st
    unet.set_attn_processor(custom_diffusion_attn_procs)
    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
    accelerator.register_for_checkpointing(custom_diffusion_layers)

    # rescale learning rate
    learning_rate = learning_rate * train_batch_size * accelerator.num_processes

    # Optimizer creation
    optimizer = torch.optim.AdamW(
        itertools.chain(text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters())
        if modifier_token is not None
        else custom_diffusion_layers.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    # Dataset and DataLoaders creation:
    concepts_list = [
        {
            "instance_prompt": instance_prompt,
            "class_prompt": None,
            "instance_data_dir": instance_data_dir,
            "class_data_dir": None,
        }
    ]
    resolution = 512

    train_dataset = CustomDiffusionDataset(
        concepts_list=concepts_list,
        tokenizer=tokenizer,
        size=resolution,
        mask_size=vae.encode(
            torch.randn(1, 3, resolution, resolution).to(dtype=weight_dtype).to(accelerator.device)
        ).latent_dist.sample().size()[-1],
        center_crop=False,
        hflip=True,
        aug=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=2,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = 0
    num_training_steps_for_scheduler = max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    if modifier_token is not None:
        custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = len(train_dataloader)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # progress bar
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        if modifier_token is not None:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                mask = batch["mask"]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
                accelerator.backward(loss)
                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if modifier_token is not None:
                    if accelerator.num_processes > 1:
                        grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                    for i in range(1, len(modifier_token_id)):
                        index_grads_to_zero = index_grads_to_zero & (
                            torch.arange(len(tokenizer)) != modifier_token_id[i]
                        )
                    grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                        index_grads_to_zero, :
                    ].fill_(0)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(text_encoder.parameters(), custom_diffusion_layers.parameters())
                        if modifier_token is not None
                        else custom_diffusion_layers.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=False)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and (global_step == max_train_steps):
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    # Save the custom diffusion layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(output_dir, safe_serialization=True)
        save_new_embed(
            text_encoder,
            modifier_token_id,
            accelerator,
            modifier_token,
            output_dir,
            safe_serialization=True,
        )

    accelerator.end_training()

def training(obj_dir: str):
    ##################### TODO: Tune hyperparameters here ##########################

    instance_prompt_dict = {
        "object-5": "photo of a <new1> metallic toy",      # for object-5
        "object-6": "photo of a <new1> plushe turtle toy", # for object-6
    }

    instance_prompt = instance_prompt_dict[obj_dir]
    instance_data_dir = f"ml2025-hw10/data/{obj_dir}" # Path to images of the object to customize
    parameter_to_train = "crossattn_kv"               # "crossattn_kv" only train the K V in cross attention. Change this to "crossattn" if you also want to train Q
    learning_rate = 1e-5
    max_train_steps = 200
    train_batch_size = 8

    ################################################################################

    ckpt_dir = f"output/{obj_dir}"                   # directory name to save checkpoints

    accelerate.notebook_launcher(train_func, args=(ckpt_dir, instance_prompt, instance_data_dir, parameter_to_train, learning_rate, max_train_steps, train_batch_size), num_processes=1)

def inference(obj_dir: str):
    ckpt_dir = f"output/{obj_dir}"
    instance_data_dir = f"ml2025-hw10/data/{obj_dir}" # Path to images of the object to customize

    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    ).to("cuda")

    state_dict = load_file(os.path.join(ckpt_dir, "pytorch_custom_diffusion_weights.safetensors"), device="cpu")
    custom_attn = pipe.unet._process_custom_diffusion(state_dict=state_dict)
    attn_procs = pipe.unet.attn_processors
    attn_procs.update(custom_attn)
    pipe.unet.set_attn_processor(attn_procs)
    pipe.unet.to(dtype = pipe.unet.dtype, device = pipe.unet.device)

    pipe.load_textual_inversion(ckpt_dir, weight_name="<new1>.safetensors")

    ##################### TODO: Tune hyperparameters here ##########################

    generate_prompt_dict = {
        "object-5": "a <new1> metallic toy in the snow", # for object-5
        "object-6": "a <new1> turtle toy on a plate",    # for object-6
    }

    guidance_scale_dict = {
        "object-5": 3.0, # for object-5
        "object-6": 6.5, # for object-6
    }

    generate_prompt = generate_prompt_dict[obj_dir]

    num_inference_steps = 100
    guidance_scale = guidance_scale_dict[obj_dir]

    ################################################################################

    obj = instance_data_dir.split("/")[-1]
    output_dir = "results"
    os.makedirs(f"{output_dir}/{obj}", exist_ok = True)

    for i in range(15):
        image = pipe(
            generate_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=1.0,
        ).images[0]
        image.save(f"{output_dir}/{obj}/{i}.jpg")

def main():
    set_seed(42)
    training("object-5")
    training("object-6")
    set_seed(42)
    inference("object-5")
    inference("object-6")


if __name__ == "__main__":
    main()
