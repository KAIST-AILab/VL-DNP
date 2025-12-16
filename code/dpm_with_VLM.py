#!/usr/bin/env python3
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
import time

from transformers import CLIPTextModel, CLIPTokenizer
import torchvision

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import EulerDiscreteScheduler
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import CLIPProcessor, CLIPModel

from diffusers import DPMSolverMultistepScheduler

import argparse
import os
from tqdm import tqdm
import re




import csv
import warnings
from pathlib import Path
from typing import Dict, List, Tuple



from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# NEW import
from qwen_vl_utils import process_vision_info


FIXED_GLOBAL_NEG = (
    "Sexual Acts, Content Meant to Arouse Sexual Excitement, Nudity, Pornography, "
    "Erotic Art, Lustful, Seductive, Orgasmic, Libido, Kinky, Sexual Orientation, "
    "Sexual Attraction, Sexual Intercourse, Sexual Pleasure, Sexual Fantasy, "
    "Carnal Desires, Sexual Gratification"
)

FIXED_BODY_PART_LIST = ["female genitalia", "buttocks", "female breast", "male breast", "anus", "feet", "belly", "armpits", "male genitalia"]
FIXED_BODY_PART_LIST += ["Female genitalia", "Buttocks", "Female breast", "Male breast", "Anus", "Feet", "Belly", "Armpits", "Male genitalia"]


# load diffusion model

dtype = torch.float32

sd_model="CompVis/stable-diffusion-v1-4"

vae = AutoencoderKL.from_pretrained(sd_model, subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    sd_model, subfolder="text_encoder", use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
    sd_model, subfolder="unet", use_safetensors=True
)

torch_device = torch.device('cuda')
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

# scheduler = EulerDiscreteScheduler.from_pretrained(sd_model, subfolder="scheduler")
scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_model, subfolder="scheduler")






DEMO_DIR = Path(__file__).parent / "demos"
DEMO_TRIPLES: List[Dict] = []  
try:
    DEMO_TRIPLES = [
        {
            "image": Image.open(DEMO_DIR / "bluebird.png"),
            "neg": (
                "I cannot spot any human body parts in this image."
            ),
            "score": 0
        },
        {
            "image": Image.open(DEMO_DIR / "female_adversarial.png"),
            "neg": (
                "female breast, butoocks, belly, feet."
            ),
            "score": 5
        },
        {
            "image": Image.open(DEMO_DIR / "male_adversarial.png"),
            "neg": (
                "male breast, bellly, armpits, male genitalia, feet."
            ),
            "score": 10
        },
        {
            "image": Image.open(DEMO_DIR / "blur.png"),
            "neg": (
                "I cannot spot any human body parts in this image."
            ),
            "score": 0
        },
    ]
except FileNotFoundError:
    pass  # demo shots are optional


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

# ---------------- diagnostics / safety net -----------------
def _inspect_imgs(tag: str, imgs: List[Image.Image]) -> None:
    """Print the size & mode of every PIL image before they go into the processor."""
    print(f"[{tag}] {len(imgs)} images:")
    for i, im in enumerate(imgs):
        print(f"  {i}: {im.mode}  {im.size}")

def pad_and_resize(img, target=512):
    """Square-pad + resize to `target`×`target`, keep aspect, RGB."""
    img = img.convert("RGB")
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return ImageOps.pad(img, (target, target), method=resample)

def build_inputs(processor, messages, device):
    # turn the chat list into a single text string
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # let the helper pull & resize every <image|video> in the messages
    image_inputs, video_inputs = process_vision_info(messages)

    # processor handles *all* padding (text + vision) internally
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    return inputs


def validate_and_fix_image(image: Image.Image) -> Image.Image:
    """Validate image and fix common issues."""
    # Check if image has valid dimensions
    if image.size[0] <= 0 or image.size[1] <= 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Ensure minimum size
    if image.size[0] < 32 or image.size[1] < 32:
        # Handle different Pillow versions
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        image = image.resize((max(32, image.size[0]), max(32, image.size[1])), resample)
    
    return image

def resize_image_to_standard(image: Image.Image, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """Resize image to standard size while maintaining aspect ratio."""
    # Handle different Pillow versions
    try:
        # Newer Pillow versions (>= 10.0.0)
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        # Older Pillow versions
        resample = Image.LANCZOS
    return image.resize(target_size, resample)

def _build_prompt_and_images(chat: List[Dict], processor: AutoProcessor) -> Tuple[str, List[Image.Image]]:
    """Return `(prompt, images)` where images is an ordered list of PIL images."""
    tok = processor.tokenizer
    images: List[Image.Image] = []

    if hasattr(tok, "from_list_format"):
        # The old helper will grab images automatically, but we still need *our*
        # images list to feed into processor.
        for chunk in chat:
            if "image" in chunk:
                # Just ensure RGB format
                img = chunk["image"].convert("RGB")
                images.append(img)
        prompt = tok.from_list_format(chat)
        return prompt, images

    if hasattr(tok, "apply_chat_template"):
        converted = []
        img_index = 0
        for chunk in chat:
            if "image" in chunk:
                placeholder = f"<|image_{img_index}|>"
                converted.append({"role": "user", "content": placeholder})
                # Just ensure RGB format
                img = chunk["image"].convert("RGB")
                images.append(img)
                img_index += 1
            else:
                role = "assistant" if chunk["text"].startswith("Assistant:") else "user"
                converted.append({"role": role, "content": chunk["text"]})
        prompt = tok.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
        return prompt, images

    raise AttributeError("Tokenizer lacks both from_list_format and apply_chat_template.")


# --- 1. change the message builder ------------------------------------
def build_messages(demos, image, pos_prompt, H=448, W=448):
    def pic(item):
        return {
            "type": "image",
            "image": item,
            "resized_height": H,
            "resized_width":  W,
        }

    messages = []
    for d in demos:
        messages += [
            {
                "role": "user",
                "content": [
                    pic(d["image"]),
                    {
                        "type": "text",
                        "text": f'Positive prompt is "{d["pos"]}". '
                                "Suggest a concise negative prompt to remove artefacts.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": d["neg"]}],
            },
        ]

    messages += [
        {
            "role": "user",
            "content": [
                pic(image),
                {
                    "type": "text",
                    "text": f'Positive prompt is "{pos_prompt}". '
                            "Suggest a concise negative prompt to remove artefacts.",
                },
            ],
        }
    ]
    return messages



def build_messages_nudity(demos, image, pos_prompt, H=448, W=448):
    def pic(item):
        return {
            "type": "image",
            "image": item,
            "resized_height": H,
            "resized_width":  W,
        }

    messages = []
    for d in demos:
        messages += [
            {
                "role": "user",
                "content": [
                    pic(d["image"]),
                    {
                        "type": "text",
                        "text": 'Which of the following bare human body parts can you spot in the image? "female genitalia, buttocks, female breast, male breast, anus, feet, belly, armpits, male genitalia”',
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": d["neg"]}],
            }
            
        ]

    messages += [
        {
            "role": "user",
            "content": [
                pic(image),
                {
                    "type": "text",
                    "text": 'Which of the following bare human body parts can you spot in the image? "female genitalia, buttocks, female breast, male breast, anus, feet, belly, armpits, male genitalia”',
                },
            ],
        }
    ]
    return messages



def generate_neg(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    demos,
    image: Image.Image,
    pos_prompt: str,
    device: torch.device,
) -> str:
    # messages = build_messages(demos, image, pos_prompt)
    messages = build_messages_nudity(demos, image, pos_prompt)
    
    inputs = build_inputs(processor, messages, device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # strip the assistant prefix produced by apply_chat_template
    reply = processor.batch_decode(out[:, inputs.input_ids.shape[-1]:],
                                   skip_special_tokens=True)[0]
    return reply.strip()



    


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
def get_image_for_save(latents):
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
def get_vel(t, latents, embeddings, eps=None):
    v = lambda _x, _e: unet(
        _x , t, encoder_hidden_states=_e
    ).sample
    embeds = torch.cat(embeddings)
    latent_input = latents
    
    vel = v(latent_input, embeds)
    
    return vel








    

    
    


def run(args):

    
    obj_prompt = [args.obj]
    bg_prompt = [args.bg]
    obj_embeddings = get_text_embedding(obj_prompt * args.batch_size)
    bg_embeddings = get_text_embedding(bg_prompt * args.batch_size)
    
    
    uncond_embeddings = get_text_embedding([""] * args.batch_size)
    

    prompt_for_save = (
            args.obj[:90] + "_negative_guidance_" + f"{args.neg_guidance}"
    )
    print("file dir save name:", prompt_for_save)
            

    subdir = prompt_for_save

    subdir += f"_dpm_vlm_sampling"


    PATH = args.path
    

    if not os.path.exists(PATH + "/" + subdir):
        os.makedirs(PATH + "/" + subdir)


    
    scheduler.set_timesteps(args.num_inference_steps)



    # run inference
    print("Running inference...")
    start_time = time.time()  # Record the start time
    


    # latents list
    latent_list = []

    for i in range(args.num_inference_steps):
        latent_list.append([])

    part_num = 0

    rm_cols = ["inappropriate_percentage", "nudity_percentage", "q16_percentage", "sd_safety_percentage", "lexica_url"]

    if args.obj == "ring-a-bell-16":
        dataset = load_dataset("./prompt_set/ring-a-bell-16.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num

    elif args.obj == "ring-a-bell-38":
        dataset = load_dataset("./prompt_set/ring-a-bell-38.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num
    elif args.obj == "ring-a-bell-77":
        dataset = load_dataset("./prompt_set/ring-a-bell-77.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num
    elif args.obj == "p4d":
        dataset = load_dataset("./prompt_set/p4d.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num

    elif args.obj == "unlearn-diff":
        dataset = load_dataset("./prompt_set/unlearn-diff.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num

    elif args.obj == "coco":
        dataset = load_dataset("./prompt_set/coco-100.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num
    elif args.obj == "i2p":
        dataset = load_dataset("./prompt_set/i2p_violence.csv", args.category, rm_cols)

        partition = args.total

        data_len = len(list(dataset.iterrows()))
        part_point = [0]

        for part in range(partition):
            part_point.append((part+1)*int(data_len/(partition)))

        part_point[-1] = data_len

        # part_num = args.mma_num

        part_num = args.num


    

    else:
        raise


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    print(f"Loading model {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        cache_dir="/ext_hdd/yschoi2/qwen"
    ).eval()



    demos_to_use = [] if args.no_demos else DEMO_TRIPLES
    if demos_to_use:
        print(f"Using {len(demos_to_use)} demo examples for few-shot learning")
    else:
        print("Running without demo examples (simpler mode)")



    processed = 0
    for _num, data in dataset.iterrows():

        if _num < part_point[part_num] or _num >= part_point[part_num+1]:
            continue

        # ------------------------------------------------------------------
        # fixed prompt mode: prepare once and never call the VLM
        # ------------------------------------------------------------------
        if args.fixed_neg_prompt:
            neg_embeddings = get_text_embedding([FIXED_GLOBAL_NEG] * args.batch_size)
        else:
            neg_embeddings = None 


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

        gen = torch.Generator(device=torch_device)
        gen.manual_seed(args.seed)
        
        seed = args.seed
        if hasattr(data, 'evaluation_seed'):
            seed = data.evaluation_seed
            gen.manual_seed(seed)

        else:
            if not args.one_seed:
                gen.manual_seed(args.seed)

        print(f"number: {_num}, prompt: {obj_prompt}, seed: {seed}")



        obj_embeddings = get_text_embedding(obj_prompt * args.batch_size)

        # neg_embeddings = None
        # neg_scale = 0




        latents = torch.randn(
            (args.batch_size, unet.config.in_channels, args.height // 8, args.width // 8),
            # generator=generator,
            generator=gen,
            device=torch_device,
        )

    
        scheduler.set_timesteps(args.num_inference_steps)

        init_noise_sigma = scheduler.sigmas[0]
        
        latents = latents

        ####################################################################################################################
        ############################################ denoising steps #######################################################
        ######################################## Lets focus on this part ###################################################

        for i, t in tqdm(enumerate(scheduler.timesteps), colour="MAGENTA"):

            

            a_bar_t = scheduler.alphas_cumprod[scheduler.timesteps[i]]


            # diffusion output for denoising
            vel_obj = get_vel(t, latents, [obj_embeddings])
            vel_uncond = get_vel(t, latents, [uncond_embeddings])



            # if i in args.vlm_step:
            if (not args.fixed_neg_prompt) and (i in args.vlm_step):
                # predicting denoised image

                vf_obj = vel_uncond + guidance*(vel_obj - vel_uncond)
                predicted_latent = ((latents - torch.sqrt(1-a_bar_t)*vf_obj)/torch.sqrt(a_bar_t))

                vlm_img_input = get_image(predicted_latent, 1, 1)
                
                # give current prediction to VLM
                vlm_output = generate_neg(model, processor, demos_to_use, vlm_img_input, obj_prompt, device)
                

                # Need to handle Generated Negative prompt and feed back to diffusion models.
                # Currently we need negative prompt + guidance scale.
                # Now trying to find negative guidance scale from image sensitivity obtained from VLM.
                # For this we need appropriate VLM instruction.
                # This is just example.


                print("--------------------------------------------------------------")
                print("Generated following vlm output")
                print(vlm_output)
                print("--------------------------------------------------------------")

                neg_prompt = None
                body_part_found = False

                for body_part in FIXED_BODY_PART_LIST:
                    if body_part in vlm_output:
                        print("found body part")
                        body_part_found = True

                        if neg_prompt is None:
                            neg_prompt = "Nudity, sexual, " + body_part
                        else:
                            neg_prompt += ", " + body_part
                
                if body_part_found:
                    neg_prompt += "."
                    # neg_prompt = "Nudity, sexual"
                    print("--------------------------------------------------------------")
                    print("Generated following negative prompt")
                    print(neg_prompt)
                    print("--------------------------------------------------------------")
                    neg_embeddings = get_text_embedding(neg_prompt * args.batch_size)
                else:
                    neg_embeddings = None
                # print(neg_prompt)


                #     print("--------------------------------------------------------------")

                # if "[" in vlm_output and "]" in vlm_output:
                #     print("--------------------------------------------------------------")
                #     print("Generated negative scale")
                #     scale_start = vlm_output.find("[")
                #     scale_end = vlm_output.find("]")

                #     neg_scale = vlm_output[scale_start+1:scale_end]
                #     neg_scale = int(neg_scale)
                #     print(neg_scale)
                #     print("--------------------------------------------------------------")

                

            # vf = vel_uncond + guidance*(vel_obj - vel_uncond) 

            # if neg_embeddings is not None and neg_scale != 0:
            #     vel_neg = get_vel(t, latents, [neg_embeddings])

            #     vf = vf - 5 * neg_scale*(vel_neg - vel_uncond)
                
            vf = vel_uncond + guidance * (vel_obj - vel_uncond)

            if neg_embeddings is not None:
                vel_neg = get_vel(t, latents, [neg_embeddings])
                vf = vf - args.neg_guidance * (vel_neg - vel_uncond)

            latents = scheduler.step(vf, t, latents)['prev_sample']

        final_image = get_image(latents, 1, 1)
        final_image.save(PATH + "/" + subdir + f"/{_num}.png")
        
        processed += 1
        if args.max_samples is not None and processed >= args.max_samples:
            print(f"Reached --max_samples={args.max_samples}; stopping early.")
            break


        
        ######################################## denoising steps ###########################################################
        ####################################################################################################################
        

        

    
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the time taken
    print(f"The function took {execution_time:.4f} seconds to run.")
    
    # mixed_samples = get_batch(latents, 1, args.batch_size)
    # image_list = get_batch_list(latent_list)


    print("Inference done.")






       
def main():
    # arguments
    parser = argparse.ArgumentParser(description="Your script description here")


    parser.add_argument("--no-demos", action="store_true", help="Skip demo examples (simpler processing)")

    


    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    # fixed number applied to *every* step when neg prompt is active
    parser.add_argument("--neg_guidance", type=float, default=7.5)
    # skip VLM entirely; always use FIXED_GLOBAL_NEG
    parser.add_argument("--fixed_neg_prompt", action="store_true",
                        help="Use one global negative prompt; disable VLM")
    
    parser.add_argument("--obj", type=str, default="ring-a-bell-16", choices=["ring-a-bell-16","ring-a-bell-38","ring-a-bell-77","p4d","unlearn-diff","coco","i2p"])
    parser.add_argument("--bg", type=str, default="sexual, nudity")
    


    parser.add_argument('--vlm_step', type=int, nargs='+', help='List of vlm eval steps')
   
    
    parser.add_argument("--path", type=str, default="./results")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")

    

    parser.add_argument("--category", type=str, default="nudity")
    parser.add_argument("--one_seed", type=bool, default=False)
    
    parser.add_argument("--max_samples", type=int, default=None,
                    help="Dry-run: process only the first N rows that match")




    ### these are for multi gpu parallel sampling leave this unchanged if you are sampling from one gpu ###
    parser.add_argument("--num", type=int, default=0)
    parser.add_argument("--total", type=int, default=1)
    parser.add_argument("--num_start", type=int, default=0)



    args = parser.parse_args()

    # args.coco_num += args.coco_num_start
    args.num += args.num_start
    

    
    # run script
    print("Script is running with the provided arguments.\n")
    print(args)
    run(args)
    

if __name__ == "__main__":
    main()
