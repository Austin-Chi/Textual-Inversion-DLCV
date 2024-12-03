import os
import numpy as np
import argparse
from packaging import version
import matplotlib.pyplot as plt

import PIL
from PIL import Image
import random

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable

from einops import rearrange
# from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from pathlib import Path

from personalized_style import PersonalizedBase, imagenet_templates_small, imagenet_dual_templates_small


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

class PreprocessImage(Dataset):
    def __init__(
        self,
        data_root,
        size=512,
        repeats=100,
        placeholder_token="<new2>",
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        center_crop=False,
        coarse_class_text=None,
        per_image_tokens=False,
    ):
        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.placeholder_token = placeholder_token
        self.coarse_class_text = coarse_class_text
        self.per_image_tokens = per_image_tokens


        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(placeholder_string)
            
        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(start_code.shape[0] * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def train_inverse(model, sampler, train_data_dir, devices, args):
    """
    Train the token embedding directly within the tokenizer.

    Args:
        model: the model to be trained
        sampler: the sampler to be used for sampling
        train_data_dir: the reference images to be used for training
        args: the arguments for training

    Returns:
        None
    """
    
    clip_embedder = model.cond_stage_model

    # # Step 1: Load the saved token embedding
    # saved_embedding = torch.load('TI/embedding_textual_inversion/emb_<new2>_140.pt').to(devices[0])  # Load and move to GPU
    
    # Step 2: Add the concept token to the tokenizer (if not already added)
    if args.concept not in clip_embedder.tokenizer.get_vocab():
        clip_embedder.tokenizer.add_tokens([args.concept])
        clip_embedder.transformer.resize_token_embeddings(len(clip_embedder.tokenizer))


    # # Add the new token to the tokenizer and resize the embedding matrix
    # clip_embedder.tokenizer.add_tokens([args.concept])
    # clip_embedder.transformer.resize_token_embeddings(len(clip_embedder.tokenizer))

    # Get the token ID of the new concept token
    new_token_id = clip_embedder.tokenizer.convert_tokens_to_ids([args.concept])[0]
    print(new_token_id)

    # Initialize with the embedding of 'dog'
    word = 'cartoon'
    corgi_token_id = clip_embedder.tokenizer.convert_tokens_to_ids([word])[0]
    # Check if the token ID corresponds to [UNK]
    if corgi_token_id == clip_embedder.tokenizer.unk_token_id:
        print(f"'{word}' is not in the tokenizer vocabulary.")
    else:
        print(f"The token ID for '{word}' is {corgi_token_id}.")
    corgi_emb = clip_embedder.transformer.get_input_embeddings().weight[corgi_token_id].detach().clone()

    # Copy the original embedding matrix (before training)
    original_embedding_matrix = clip_embedder.transformer.get_input_embeddings().weight.detach().clone()
    print(original_embedding_matrix.size())
    # Assign the 'dog' embedding to the new token
    with torch.no_grad():
        clip_embedder.transformer.get_input_embeddings().weight[new_token_id] = corgi_emb.clone()

    # No separate `emb`, we directly optimize the token embedding in the model
    embedding_layer = clip_embedder.transformer.get_input_embeddings()

    # Optimizer for the embedding layer (including the new token)
    opt = torch.optim.Adam(embedding_layer.parameters(), lr=args.lr)
    # Add a learning rate scheduler (StepLR as an example)
    # scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs
    scheduler = lr_scheduler.LinearLR(
        optimizer=opt,  # Your optimizer here
        start_factor=1.0,  # Start with the full learning rate
        end_factor=0.0,    # Linearly reduce to zero
        total_iters=args.epochs  # Number of epochs for linear decay
    )

    # Dataset and DataLoader setup
    train_dataset = PreprocessImage(
        data_root=train_data_dir,
        size=args.resolution,
        repeats=args.repeats,
        placeholder_token=args.concept,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )    
    
    fixed_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                cond, args.image_size, args.image_size, args.ddim_steps, s, args.ddim_eta,
                                                                start_code=code, till_T=t, verbose=False)
    
    def decode_and_save_image(model, z, path):
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0)/2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b h w c')
        image = Image.fromarray((x[0].cpu().numpy()*255).astype(np.uint8))
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path)
        plt.close()
    
    os.makedirs('evaluation_folder', exist_ok=True)
    os.makedirs('evaluation_folder/textual_inversion', exist_ok=True)
    os.makedirs(f'evaluation_folder/textual_inversion/{args.concept}', exist_ok=True)
    os.makedirs(f'{args.models_path}/embedding_textual_inversion', exist_ok=True)

    # Training loop
    for epoch in range(150, 150+args.epochs):
        for i, batch in enumerate(train_dataloader):
            opt.zero_grad()
            model.zero_grad()
            model.train()

            # Convert images to latent space
            batch_images = batch['pixel_values'].to(devices[0])
            encoder_posterior = model.encode_first_stage(batch_images)
            batch_z = model.get_first_stage_encoding(encoder_posterior).detach()

            # Get the learned conditioning embedding from the tokenizer
            batch_prompt = batch['caption']
            # emb_prompt = model.get_learned_conditioning([f"a photo of {args.concept} style"])
            emb_prompt = model.get_learned_conditioning(batch_prompt)

            # Use the embedding as conditioning for the diffusion model
            cond = torch.repeat_interleave(emb_prompt, batch_z.shape[0], dim=0)

            # random timestep
            t_enc = torch.randint(0, args.ddim_steps, (1,), device=devices[0]).long()

            # time step from 1000 to 0 (0 being good)
            og_num = round((int(t_enc)/args.ddim_steps)*1000)
            og_num_lim = round((int(t_enc+1)/args.ddim_steps)*1000)

            t_enc_ddpm = torch.randint(og_num, og_num_lim, (batch_z.shape[0],), device=devices[0])


            # Add noise and run forward/backward diffusion steps
            noise = torch.randn_like(batch_z) * args.noise_scale
            x_noisy = model.q_sample(x_start=batch_z, t=t_enc_ddpm, noise=noise)
            # model_output = model.apply_model(x_noisy, t_enc_ddpm, cond)
            model_output = model.apply_model(x_noisy, t_enc_ddpm, emb_prompt)
            
            loss = torch.nn.functional.mse_loss(model_output, noise)
            loss.backward()
            opt.step()

            if i % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
        scheduler.step()
        # Inference with the learned embedding
        with torch.no_grad():
            token_embedding = clip_embedder.transformer.get_input_embeddings().weight[new_token_id].detach().clone()
            clip_embedder.transformer.get_input_embeddings().weight[:-1] = original_embedding_matrix[:-1].clone()
            # clip_embedder.transformer.get_input_embeddings().weight[new_token_id] = token_embedding.detach().clone()

            model.eval()
            # Reconstruct an image using the learned token embedding
            emb_val = model.get_learned_conditioning([f"The streets of Paris in the style of {args.concept}."])
            z_r_till_T = quick_sample_till_t(emb_val.to(devices[0]), args.start_guidance, fixed_start_code, int(args.ddim_steps))
            decode_and_save_image(model, z_r_till_T, path=f'evaluation_folder/textual_inversion/{args.concept}/gen_{epoch}.png')
            emb_org = model.get_learned_conditioning([f"a picture in the style of {args.concept}."])
            z_r_till_T_org = quick_sample_till_t(emb_org.to(devices[0]), args.start_guidance, fixed_start_code, int(args.ddim_steps))
            decode_and_save_image(model, z_r_till_T_org, path=f'evaluation_folder/textual_inversion/{args.concept}/gen_{epoch}_org.png')
            emb_mia = model.get_learned_conditioning([f"a photo of Miyazaki style."])
            z_r_till_T_mia = quick_sample_till_t(emb_mia.to(devices[0]), args.start_guidance, fixed_start_code, int(args.ddim_steps))
            decode_and_save_image(model, z_r_till_T_mia, path=f'evaluation_folder/textual_inversion/{args.concept}/gen_{epoch}_mia.png')

            # Save the embedding after each epoch
            torch.save(clip_embedder.transformer.get_input_embeddings().weight[new_token_id].detach().cpu(), 
                       f'{args.models_path}/embedding_textual_inversion/emb_{args.concept}_{epoch}.pt')
            
    return clip_embedder.transformer.get_input_embeddings().weight[new_token_id].detach()


def personalized_inference(model, sampler, devices, args, output_dir, prompt, ind=0):
    """
    Perform inference using the learned embedding for personalized image generation.
    
    Args:
        model: the pretrained diffusion model
        sampler: the sampler for generating samples
        devices: the list of GPU devices to use
        args: the arguments containing model paths and parameters
        output_dir: the folder to save generated images
        prompt: the text prompt containing the learned token (e.g., "A <new1> on the beach at sunset")

    Returns:
        None
    """

    # Load the learned embedding
    
    # clip_embedder = FrozenCLIPEmbedder(device="cuda")
    # clip_embedder = model.cond_stage_model
    
    emb_path = f'{args.models_path}/embedding_textual_inversion/emb_{args.concept}_99.pt'
    learned_emb = torch.load(emb_path).to(devices[0])
    print(learned_emb.size())

    # Get the initial conditioning for the prompt
    
    # get dictionary
    # add 
    # print (len(clip_embedder.tokenizer))
    # clip_embedder.tokenizer.add_tokens(['<new1>', '<new2>'])
    # print (len(clip_embedder.tokenizer))
    
    # clip_embedder.transformer.resize_token_embeddings(len(clip_embedder.tokenizer))
    
    emb = model.get_learned_conditioning([prompt])
    # encoder_posterior = model.encode_first_stage(prompt) #prompt
    # emb = model.get_first_stage_encoding(encoder_posterior).detach()
    
    # Replace the token-specific embedding in the prompt with the learned embedding
    # Find the index of the learned token in the prompt and replace that part of the embedding
    token_name = args.concept  # for example, "<new1>"
    print(emb.size())
    if token_name in prompt:
        token_idx = prompt.split().index(token_name)
        emb[0, 4:6, :] = learned_emb[0, 1:3, :]

    # Generate an image using the personalized prompt
    fixed_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])  # You can change this starting code if needed

    with torch.no_grad():
        z_r_till_T = sample_model(model, sampler,
                                  emb, args.image_size, args.image_size, args.ddim_steps, args.start_guidance, 
                                  args.ddim_eta, start_code=fixed_start_code, verbose=False)
        
        # Decode the latent vector and save the image
        x = model.decode_first_stage(z_r_till_T)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)  # Ensure pixel values are between 0 and 1
        x = rearrange(x, 'b c h w -> b h w c')
        
        # Convert to a PIL image and save
        image = Image.fromarray((x[0].cpu().numpy() * 255).astype(np.uint8))
        os.makedirs(output_dir, exist_ok=True)
        image.save(f'{output_dir}/personalized_image_{ind}.png')
        print(f'Personalized image saved at: {output_dir}/personalized_image.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Textual Inversion Training Script')
    parser.add_argument('--concept', help='Concept to be learned', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=5e-4)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--ddim_eta', help='ddim eta used to train', type=float, required=False, default=0.0)
    parser.add_argument('--info', help='info to add to model name', type=str, required=False, default='')

    parser.add_argument('--models_path', help='path to save output model', type=str, required=True, default='models')
    parser.add_argument('--train_data_dir', help='path to training data', type=str, required=True)
    parser.add_argument('--resolution', help='resolution of training data', type=int, required=False, default=512)
    parser.add_argument('--repeats', help='number of repeats of training data', type=int, required=False, default=100)
    parser.add_argument('--center_crop', help='whether to center crop training data', type=bool, required=False, default=False)
    parser.add_argument('--train_batch_size', help='batch size for training', type=int, required=False, default=4)
    parser.add_argument('--dataloader_num_workers', help='number of workers for dataloader', type=int, required=False, default=4)
    parser.add_argument('--epochs', help='number of epochs to train', type=int, required=False, default=100)
    parser.add_argument('--noise_scale', help='noise scale for training', type=float, required=False, default=1.0)
    parser.add_argument('--verbose', help='whether to print verbose', type=bool, required=False, default=True)


    args = parser.parse_args()
    
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]

    # load model and sampler
    model = load_model_from_config(args.config_path, args.ckpt_path, devices[0])
    # sampler = DDIMSampler(model)
    sampler = DPMSolverSampler(model)

    # train the inverse model
    emb = train_inverse(model, sampler, args.train_data_dir, devices, args)

    # save the learned embedding
    os.makedirs(f'{args.models_path}/embedding_textual_inversion', exist_ok=True)
    torch.save(emb, f'{args.models_path}/embedding_textual_inversion/emb_{args.concept}.pt')

    # personalized_inference(model, sampler, devices, args, "generated_images", "A photo of a <new1> perched on a park bench with the Colosseum looming behind.")

