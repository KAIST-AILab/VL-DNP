import torch
import numpy as np
import pandas as pd
import time

from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import EulerDiscreteScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import CLIPProcessor, CLIPModel

from diffusers import DPMSolverMultistepScheduler
from diffusers import DDIMScheduler
from diffusers import DDPMScheduler


import wandb
import argparse
import os
from tqdm import tqdm
import re

PATH = "/ext2/yschoi2/results_super_diff/"

dtype = torch.float32
device = torch.device("cuda:0")

sd_model="CompVis/stable-diffusion-v1-4"

vae = AutoencoderKL.from_pretrained(sd_model, subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    sd_model, subfolder="text_encoder", use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
    sd_model, subfolder="unet", use_safetensors=True
)

torch_device = torch.device('cuda:0')
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

# scheduler = EulerDiscreteScheduler.from_pretrained(sd_model, subfolder="scheduler")
scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_model, subfolder="scheduler")
# scheduler = DDIMS cheduler.from_pretrained(sd_model, subfolder="scheduler")
# scheduler = DDPMScheduler.from_pretrained(sd_model, subfolder="scheduler")

def load_dataset(data_f, category, rm_cols, debug=False):
    # Read CSV file
    dataset = pd.read_csv(data_f)
    # delete index column
    
    if 'Unnamed: 0' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 0'])  
    dataset["nudity_toxicity"] = 0.0
    
    print(f"{category} dataset size: {dataset.shape[0]}")
    return dataset




@torch.no_grad
def get_image(latents, nrow, ncol):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    if len(image.shape) < 4:
        image = image.unsqueeze(0)
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    rows = []
    for row_i in range(nrow):
        row = []
        for col_i in range(ncol):
            i = row_i * nrow + col_i
            row.append(image[i])
        rows.append(torch.hstack(row))
    image = torch.vstack(rows)
    return Image.fromarray(image.cpu().numpy())

@torch.no_grad
def get_image_for_save(latents, nrow, ncol):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = image.permute(0, 2, 3, 1) * 255  # .to(torch.uint8)
    return image

@torch.no_grad
def get_batch(latents):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    if len(image.shape) < 4:
        image = image.unsqueeze(0)
    image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    return image

@torch.no_grad
def get_batch_list(latents):
    image_list = []
    for num in range(len(latents)):
        image = vae.decode(latents[num].unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)

        image_list.append(image)

    image_list = torch.cat(image_list)
    return image_list

@torch.no_grad
def get_text_embedding(prompt):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(text_input.input_ids.to(torch_device))[0]

@torch.no_grad
def get_vel(t, sigma, latents, embeddings, eps=None, get_div=False):
    v = lambda _x, _e: unet(
        _x , t, encoder_hidden_states=_e
    ).sample
    embeds = torch.cat(embeddings)
    latent_input = latents
    if get_div:
        with sdpa_kernel(SDPBackend.MATH):
            vel, div = torch.func.jvp(
                v, (latent_input, embeds), (eps, torch.zeros_like(embeds))
            )
            div = -(eps * div).sum((1, 2, 3))
    else:
        vel = v(latent_input, embeds)
        div = torch.zeros([len(embeds)], device=torch_device)
    return vel, div


def compute_clip_score(clip_processor, clip, mixed_samples, args):
    score_min, score_avg, raw_scores = [], [], []
    with torch.no_grad():
        for i in range(args.batch_size):
            inputs = clip_processor(
                text=[args.obj],
                images=mixed_samples[i].unsqueeze(0),
                return_tensors="pt",
                padding=True,
            )
            outputs = clip(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            sim_sd_A = logits_per_image.cpu().item()

            inputs = clip_processor(
                text=[args.bg],
                images=mixed_samples[i].unsqueeze(0),
                return_tensors="pt",
                padding=True,
            )

            # outputs = clip(**inputs)
            # logits_per_image = (
            #     outputs.logits_per_image
            # )  # this is the image-text similarity score
            sim_sd_B = logits_per_image.cpu().item()

            # score_min.append(min(sim_sd_A, sim_sd_B))
            # score_avg.append((sim_sd_A + sim_sd_B) / 2)
            raw_scores.append((sim_sd_A, sim_sd_B))


            score_min.append(sim_sd_A)
            score_avg.append(sim_sd_A)
            # raw_scores.append(sim_sd_A, sim_sd_B)
            
    return score_min, score_avg, raw_scores

def compute_clip_score_list_partial(clip_processor, clip, mixed_samples, prompt_list, part_point, part_num):
    score_avg, raw_scores = [], []

    with torch.no_grad():
        # for i in range(args.batch_size):
        for num, data in prompt_list.iterrows():
            # if num == len(mixed_samples):
            #     break
            
            if num < part_point[part_num] or num >= part_point[part_num+1]:
                continue

            if "adv_prompt" in data:
                image_prompt = data['adv_prompt']
                # case_num = _iter
            # Concept removal
            elif "sensitive prompt" in data:
                image_prompt = data["sensitive prompt"]
                # case_num = _iter
            elif "prompt" in data:
                image_prompt = data["prompt"]


            inputs = clip_processor(
                text=[image_prompt],
                images=mixed_samples[num - part_point[part_num]].unsqueeze(0),
                return_tensors="pt",
                padding=True,
            )
            outputs = clip(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            sim_sd_A = logits_per_image.cpu().item()

            raw_scores.append(sim_sd_A)

            score_avg.append(sim_sd_A)

            
            
    return score_avg, raw_scores

def compute_clip_score_list(clip_processor, clip, mixed_samples, prompt_list, part_point, part_num):
    score_avg, raw_scores = [], []
    with torch.no_grad():
        # for i in range(args.batch_size):
        for num, data in prompt_list.iterrows():
            if num == len(mixed_samples):
                break
            


            inputs = clip_processor(
                text=[data[0]],
                images=mixed_samples[num].unsqueeze(0),
                return_tensors="pt",
                padding=True,
            )
            outputs = clip(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            sim_sd_A = logits_per_image.cpu().item()

            raw_scores.append(sim_sd_A)

            score_avg.append(sim_sd_A)

            
            
    return score_avg, raw_scores


def compute_image_reward(image_reward, mixed_samples, args):
    score_min, score_avg, raw_scores = [], [], []
    with torch.no_grad():
        for i in range(args.batch_size):
            img = mixed_samples[i].unsqueeze(0)
            img = Image.fromarray(img.squeeze(0).cpu().numpy())
            
            rewards_A = image_reward.score(args.obj, img)
            rewards_B = image_reward.score(args.bg, img)
            
            # score_min.append(min(rewards_A, rewards_B))
            # score_avg.append((rewards_A + rewards_B) / 2)
            raw_scores.append((rewards_A, rewards_B))
            
            score_min.append(rewards_A)
            score_avg.append(rewards_A)
            # raw_scores.append(rewards_A)

    return score_min, score_avg, raw_scores
    

    
    


def run(args):
    
    lift = 0.0
    kappa = 0.5*torch.ones((args.num_inference_steps+1,args.batch_size), device=torch_device)
    obj_prompt = [args.obj]
    bg_prompt = [args.bg]
    obj_embeddings = get_text_embedding(obj_prompt * args.batch_size)
    bg_embeddings = get_text_embedding(bg_prompt * args.batch_size)
    
    
    uncond_embeddings = get_text_embedding([""] * args.batch_size)
    

    prompt_for_save = (
            args.obj[:90] + "_" + "and" + "_" + re.sub(r'[^A-Za-z0-9 ]+', '', args.bg.replace(" ", "_")[:90])
        )
    print("file dir save name:", prompt_for_save)
            


    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    
    scheduler.set_timesteps(args.num_inference_steps)



    # run inference
    print("Running inference...")
    start_time = time.time()  # Record the start time

    

    if args.safe_diff:
        
        PATH = f"/ext_hdd/yschoi2/gen_ai_project/vanilla_dpm_ori_pred/"
    elif args.comp_glide:
        PATH = "/ext_hdd/yschoi2/results_super_diff/composed_glide/"
    else:
        PATH = "/ext_hdd/yschoi2/results_super_diff/vanilla/"

    print(f"safe_diffusion?: {args.safe_diff}")
    print(f"composed_glide?: {args.comp_glide}")
    print(f"glide_w: {args.glide_w}")
    if args.path is None:
        print(f"path: {PATH}")
    else:
        print(f"path: {args.path}")

    



    # prediction
    prediction_list = []

    # latents list
    latent_list = []

    for i in range(args.num_inference_steps):
        latent_list.append([])

    part_num = 0

    rm_cols = ["inappropriate_percentage", "nudity_percentage", "q16_percentage", "sd_safety_percentage", "lexica_url"]

    if args.obj == "ring-a-bell-16":
        dataset = load_dataset("./prompt_set/Nudity_eta_3_K_16.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num

    elif args.obj == "ring-a-bell-38":
        dataset = load_dataset("./prompt_set/Nudity_eta_3_K_38.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num
    elif args.obj == "ring-a-bell-77":
        dataset = load_dataset("./prompt_set/Nudity_eta_3_K_77.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num


    elif args.obj == "p4d":
        dataset = load_dataset("./prompt_set/p4dn_16_prompt.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num


    elif args.obj == "mma-diff":
        dataset = load_dataset("./prompt_set/mma-diffusion.csv", args.category, rm_cols)
        # partition = args.mma_total

        partition = args.total

        data_len = len(list(dataset.iterrows()))

        part_point = [0]
        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition) + 1))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num


    elif args.obj == "unlearn-diff":
        dataset = load_dataset("./prompt_set/unlearn-diffusion.csv", args.category, rm_cols)
        # partition = args.unlearn_total

        partition = args.total

        data_len = len(list(dataset.iterrows()))

        part_point = [0]
        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition) + 1))

        part_point[-1] = data_len

        # part_num = args.unlearn_num

        part_num = args.num

    elif args.obj == "coco-100":
        dataset = load_dataset("./prompt_set/coco-100.csv", args.category, rm_cols)

        # partition = args.coco_total

        partition = args.total

        data_len = len(list(dataset.iterrows()))

        part_point = [0]
        for part in range(partition):
            part_point.append(   (part+1) * (int(data_len/(partition)) + 1) )

        part_point[-1] = data_len

        # part_num = args.coco_num

        part_num = args.num

    elif args.obj == "i2p-violence":
        dataset = load_dataset("./prompt_set/i2p_violence.csv", args.category, rm_cols)

        # partition = args.coco_total

        partition = args.total

        data_len = len(list(dataset.iterrows()))

        part_point = [0]
        for part in range(partition):
            part_point.append(   (part+1) * (int(data_len/(partition)) + 1) )

        part_point[-1] = data_len

        # part_num = args.coco_num

        part_num = args.num


    else:
        raise


    for _num, data in dataset.iterrows():

        if _num < part_point[part_num] or _num >= part_point[part_num+1]:
            continue



        if "adv_prompt" in data:
            obj_prompt = data['adv_prompt']
            # case_num = _iter
        # Concept removal
        elif "sensitive prompt" in data:
            obj_prompt = data["sensitive prompt"]
            # case_num = _iter
        elif "prompt" in data:
            obj_prompt = data["prompt"]
            # case_num = data["case_number"]

        if hasattr(data, 'guidance'):
            guidance = data.guidance
        elif hasattr(data, 'evaluation_guidance'):
            guidance = data.evaluation_guidance
        elif hasattr(data, 'sd_guidance_scale'):
            guidance = data.sd_guidance_scale
        else:
            guidance = 7.5


        
        seed = args.seed
        if hasattr(data, 'evaluation_seed'):
            seed = data.evaluation_seed
            # generator = torch.cuda.manual_seed(seed)
            gen.manual_seed(seed)

        else:
            # generator = torch.cuda.manual_seed(args.seed)
            if not args.one_seed:
                gen.manual_seed(args.seed)

        print(f"number: {_num}, prompt: {obj_prompt}, seed: {seed}")
        # obj_prompt = [data[0]]
       
        # bg_prompt = [args.bg]
        obj_embeddings = get_text_embedding(obj_prompt * args.batch_size)
        # bg_embeddings = get_text_embedding(bg_prompt * args.batch_size)

        ll_obj = torch.ones((args.num_inference_steps+1,args.batch_size), device=torch_device)
        ll_bg = torch.ones((args.num_inference_steps+1,args.batch_size), device=torch_device)
        ll_uncond = torch.ones((args.num_inference_steps+1,args.batch_size), device=torch_device)



        latents = torch.randn(
            (args.batch_size, unet.config.in_channels, args.height // 8, args.width // 8),
            # generator=generator,
            generator=gen,
            device=torch_device,
        )

    
        scheduler.set_timesteps(args.num_inference_steps)

        init_noise_sigma = scheduler.sigmas[0]
        
        latents = latents



        for i, t in tqdm(enumerate(scheduler.timesteps), colour="MAGENTA"):

            dsigma = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
            sigma = scheduler.sigmas[i]
            vel_obj, _ = get_vel(t, sigma, latents, [obj_embeddings])
            vel_uncond, _ = get_vel(t, sigma, latents, [uncond_embeddings])

                

            vf = vel_uncond + guidance*(vel_obj - vel_uncond) 
                        


            a_bar_t = scheduler.alphas_cumprod[scheduler.timesteps[i]]
            latent_list[i].append((latents - torch.sqrt(1-a_bar_t)*vf)/torch.sqrt(a_bar_t))


            latents = scheduler.step(vf, t, latents)['prev_sample']

            

            wandb.log({"iter": i})


            
        # latent_list.append(latents)
        
        
    subdir = prompt_for_save

    subdir += f"{args.seed}_total_step_{args.num_inference_steps}_vanilla_sampling"


    if args.path is not None:
        PATH = args.path
    

    if not os.path.exists(PATH + "/" + subdir):
        os.makedirs(PATH + "/" + subdir)

    
    
    
    

    # latent_list = torch.cat(latent_list, dim=0)

    data_size = len(latent_list)

    for i in range(args.num_inference_steps):
        
        if not os.path.exists(PATH + "/" + subdir +  f"/timestep_{i}"):
            os.makedirs(PATH + "/" + subdir +  f"/timestep_{i}")

        temp_latent_batch = torch.cat(latent_list[i], dim=0)
        torch.save(temp_latent_batch, PATH + "/" + subdir +  f"/timestep_{i}"   +f"/timestep_{i}_latent_list.pt" )

        for _num, latent in enumerate(temp_latent_batch):
            img_to_save = get_image(temp_latent_batch[_num].unsqueeze(0), 1, 1)
            img_to_save.save(PATH + "/" + subdir +  f"/timestep_{i}"   +f"/timestep_{i}_{_num}.png")

        temp_image_list = get_batch_list(temp_latent_batch)

        torch.save(temp_image_list, PATH + "/" + subdir +  f"/timestep_{i}"   +f"/timestep_{i}_image_list.pt" )

    
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the time taken
    print(f"The function took {execution_time:.4f} seconds to run.")
    
    # mixed_samples = get_batch(latents, 1, args.batch_size)
    # image_list = get_batch_list(latent_list)


    print("Inference done.")






       
def main():
    # arguments
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--T", type=float, default=1, help="temperature for OR")
    
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--obj", type=str, default="ring-a-bell-16")
    parser.add_argument("--bg", type=str, default="a dog")
    

   
    
    parser.add_argument("--path", type=str, default=None)


    parser.add_argument("--safe_diff", type=bool, default=False)
    parser.add_argument("--comp_glide", type=bool, default=False)
    parser.add_argument("--glide_w", type=float, default=0.3)
    

    # parser.add_argument("--prompt_path", type=str, default="./prompt_set/nudity-ring-a-bell.csv")
    parser.add_argument("--category", type=str, default="nudity")




    parser.add_argument("--one_seed", type=bool, default=False)




    ### these are for multi gpu parallel sampling leave this unchanged if you are sampling from one gpu ###
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--total", type=int, default=1)
    parser.add_argument("--num_start", type=int, default=0)



    args = parser.parse_args()

    # args.coco_num += args.coco_num_start
    args.num += args.num_start
    
    wandb.init(
        project="superdiff_imgs", 
        config=args,
        entity='yoonseok97',
        name = args.obj[:70],
        dir = "/ext_hdd/yschoi2/wandb"
        )
    
    # run script
    print("Script is running with the provided arguments.\n")
    print(args)
    run(args)
    

if __name__ == "__main__":
    main()
