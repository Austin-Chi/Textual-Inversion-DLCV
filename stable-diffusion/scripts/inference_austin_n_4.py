import os
import json
import numpy as np
import argparse
from packaging import version
import matplotlib.pyplot as plt
import time
import PIL
from PIL import Image

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from einops import rearrange
# from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from pathlib import Path
# from grade_hw2_3 import evaluate



# Util Functions

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


def load_and_generate_image(model, sampler, device, args, prompts, output_path, repeats=25, seed=4):
    seed_everything(seed)
    batch_size = args.n_samples
    for prompt_idx, prompt in enumerate(prompts):
        assert prompt is not None
        data = [batch_size * [prompt]]
        
        img_folder = os.path.join(output_path, str(prompt_idx))
        os.makedirs(img_folder, exist_ok=True)
        sample_path = img_folder
        os.makedirs(sample_path, exist_ok=True)
        base_count = 0

        start_code = None
        if args.fixed_code:
            start_code = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)

        precision_scope = autocast if args.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in range(args.n_iter):
                        for prompts in data:
                            uc = None
                            if args.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])#.cpu()
                            
                            c = model.get_learned_conditioning(prompts)
                            
                            shape = [args.C, args.H // args.f, args.W // args.f]
                            samples_ddim, _ = sampler.sample(
                                S=args.ddim_steps,
                                conditioning=c,
                                batch_size=args.n_samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=args.scale,
                                unconditional_conditioning=uc,
                                eta=args.ddim_eta,
                                x_T=start_code
                            )
                            
                            # Free up conditioning tensors after use
                            del c, uc
                            torch.cuda.empty_cache()
                            
                            # Decode and save the image
                            x_samples_ddim = model.decode_first_stage(samples_ddim).cpu()
                            x_samples_ddim = torch.clamp((x_samples_ddim.float() + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1).numpy()

                            # Convert and save each image in the batch
                            x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                if base_count < repeats:
                                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                            # Free samples from memory after each batch
                            del samples_ddim, x_samples_ddim, x_checked_image_torch
                            torch.cuda.empty_cache()

        print(f"Your samples are ready and waiting for you here: \n{img_folder}\nEnjoy.")


def load_json_and_generate(json_path, model, sampler, device, args, output_path, repeats = 25, seed = 0):
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    os.makedirs(output_path, exist_ok=True)

    for key, value in data.items():
        inf_folder = os.path.join(output_path, key)
        os.makedirs(inf_folder, exist_ok=True)
        token = value['token_name']
        seed = 6
        if token == '<new2>':
            seed = 12
        prompts = value['prompt']
        load_and_generate_image(model, sampler, device, args, prompts, inf_folder, repeats, seed)


if __name__ == '__main__':
    print ('=======start========')
    start_time = time.time()
    parser = argparse.ArgumentParser(description = 'Textual Inversion Training Script')
    # 使用位置參數來傳入 json_path, output_path, ckpt_path_2
    parser.add_argument('json_path', help='path to the json input', type=str)
    parser.add_argument('output_path', help='path to the output folder', type=str)
    parser.add_argument('ckpt_path_2', help='path to the second .ckpt model file', type=str)
    
    # parser.add_argument('--concept', help='Concept to be learned', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=5e-4)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
    # parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--ddim_eta', help='ddim eta used to train', type=float, required=False, default=0.0)
    parser.add_argument('--info', help='info to add to model name', type=str, required=False, default='')
    parser.add_argument('--fixed_code', action='store_true', help="if enabled, uses the same starting code across samples ",)
    parser.add_argument('--n_samples', type=int, default=5, help="how many samples to produce for each given prompt. A.k.a. batch size",)
    parser.add_argument('--n_iter', type=int, default=5, help="sample this often",)
    parser.add_argument('--H', type=int, default=512, help="image height, in pixel space",)
    parser.add_argument('--W', type=int, default=512, help="image width, in pixel space",)
    parser.add_argument('--C', type=int, default=4, help="latent channels",)
    parser.add_argument('--f', type=int, default=8, help="downsampling factor",)
    parser.add_argument( '--scale', type=float, default=7.5, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",)
    parser.add_argument('--precision', type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")

    # parser.add_argument('--models_path', help='path to save output model', type=str, required=True, default='models')
    # parser.add_argument('--train_data_dir', help='path to training data', type=str, required=True)
    parser.add_argument('--resolution', help='resolution of training data', type=int, required=False, default=512)
    parser.add_argument('--repeats', help='number of repeats of training data', type=int, required=False, default=100)
    parser.add_argument('--center_crop', help='whether to center crop training data', type=bool, required=False, default=False)
    parser.add_argument('--train_batch_size', help='batch size for training', type=int, required=False, default=4)
    parser.add_argument('--dataloader_num_workers', help='number of workers for dataloader', type=int, required=False, default=4)
    parser.add_argument('--epochs', help='number of epochs to train', type=int, required=False, default=100)
    # parser.add_argument('--noise_scale', help='noise scale for training', type=float, required=False, default=1.0)
    parser.add_argument('--verbose', help='whether to print verbose', type=bool, required=False, default=True)
    # parser.add_argument('--json_path', help='path to the json input', type=str, required=True, default='../hw2_data/textual_inversion/input.json')
    # parser.add_argument('--output_path', help='path to the output folder', type=str, required=True, default='output_austin')
    

    args = parser.parse_args()
    seed_everything(3)
    # devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    

    # load model and sampler
    config = OmegaConf.load(f"{args.config_path}")
    model = load_model_from_config(config, args.ckpt_path_2, device)
    # sampler = DDIMSampler(model)
    sampler = DPMSolverSampler(model)
    # new1 embedding path
    # embedding_path_1 = 'TI/embedding_textual_inversion/emb_<new1>_251.pt'#169 pass 1
    embedding_path_1 = 'DLCV_hw2_models/p3_emb_<new1>.pt'
    # embedding_path_1 = 'TI/embedding_textual_inversion/emb_<new1>_295.pt'
    special_token_1 = '<new1>'
    # Step 1: Load the saved token embedding
    saved_embedding_1 = torch.load(embedding_path_1).to(device)  # Load and move to GPU
    
    # Step 2: Add the concept token to the tokenizer (if not already added)
    clip_embedder = model.cond_stage_model
    if special_token_1 not in clip_embedder.tokenizer.get_vocab():
        clip_embedder.tokenizer.add_tokens([special_token_1])
        clip_embedder.transformer.resize_token_embeddings(len(clip_embedder.tokenizer))

    
    # Step 3: Get the token ID for the concept and replace its embedding with the loaded one
    new_token_id_1 = clip_embedder.tokenizer.convert_tokens_to_ids([special_token_1])[0]
    
    with torch.no_grad():
        clip_embedder.transformer.get_input_embeddings().weight[new_token_id_1] = saved_embedding_1
    
    # Free GPU memory for saved_embedding_1
    saved_embedding_1 = saved_embedding_1.to('cpu')  # Move back to CPU
    del saved_embedding_1  # Delete the variable
    torch.cuda.empty_cache()  # Clear GPU cache

    # new2 embedding path
    # embedding_path_2 = 'TI/embedding_textual_inversion/emb_<new2>_349.pt' #349 1 pass, 2 txt fail
    embedding_path_2 = 'DLCV_hw2_models/p3_emb_<new2>.pt'
    # embedding_path_2 = 'TI/embedding_textual_inversion/emb_<new2>_342.pt'
    special_token_2 = '<new2>'
    # Step 1: Load the saved token embedding
    saved_embedding_2 = torch.load(embedding_path_2).to(device)  # Load and move to GPU
    
    # Step 2: Add the concept token to the tokenizer (if not already added)
    clip_embedder = model.cond_stage_model
    if special_token_2 not in clip_embedder.tokenizer.get_vocab():
        clip_embedder.tokenizer.add_tokens([special_token_2])
        clip_embedder.transformer.resize_token_embeddings(len(clip_embedder.tokenizer))

    
    # Step 3: Get the token ID for the concept and replace its embedding with the loaded one
    new_token_id_2 = clip_embedder.tokenizer.convert_tokens_to_ids([special_token_2])[0]
    
    with torch.no_grad():
        clip_embedder.transformer.get_input_embeddings().weight[new_token_id_2] = saved_embedding_2
    
    # Free GPU memory for saved_embedding_2
    saved_embedding_2 = saved_embedding_2.to('cpu')
    del saved_embedding_2
    torch.cuda.empty_cache()
    # for i in range(1, 100):
    i = 0
    load_json_and_generate(args.json_path, model, sampler, device, args, args.output_path, 25, i)
    print(f"Images saved to {args.output_path}!Enjoy!")
    # input_dir = '../hw2_data/textual_inversion/'
    # evaluate(args.json_path, input_dir, args.output_path, i,log_file="evaluation_austin_last_try.log")
    # print(f"evaluation for {i} success!!")

    end_time = time.time()
        # 計算總共花費的時間
    elapsed_time = end_time - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds")