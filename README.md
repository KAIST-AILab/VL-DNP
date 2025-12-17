## VL-DNP

## 1  Set-up

```bash
# create / activate your venv or conda env first
pip install -r requirements.txt

#  (Optional) install a GPU wheel if you have CUDA 12.1
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
````

The pinned versions avoid the `numpy 2.0` / `transformers` padding bug that
throws **“expected np.ndarray (got numpy.ndarray)”**.

---



## 2 Running Diffusion Model with VLM

VL-DNP is based on Qwen 2.5-VL-7B-Instruct and Stable Diffusion v1.4

```bash
# VLM evaluating during Diffusion sampling.
python code/dpm_with_VLM.py \
	--path results \
	--vlm_step 5 6 7 9 12 16 21 27 34 42 \
	--obj ring-a-bell-16 \
	--neg_guidance 15
```

* `--path`	directory to save generated image.
* `--vlm_step`	steps that VLM generates negative prompt.
* `--obj`	evaluating prompt set. coco is for normal prompts. ring-a-bell is for adversarial prompts.
* `--neg_guidance`	negative guidance scale to be used


---

## 3 Running Diffusion Model with Static Negative Prompting

```bash
# negative prompting evaluation using various negative guidance sclae
python code/negative_prompt.py \
	--path results_neg_prompt \
	--neg_guidance 15 \
	--obj ring-a-bell-16
```

* `--path`	directory to save generated image.
* `--obj`	evaluating prompt set. coco is for normal prompts. ring-a-bell is for adversarial prompts.
* `--neg_guidance`	negative guidance scale to be used

---

## 4 Evaluation with Nudenet Classifier (from SAFREE)
You can download Classifier model at [Nudenet Classifier](https://github.com/notai-tech/nudenet)

Download and place at 'classifier/'

Evaluation can be done by

```bash
# Nudity evaluation using Nudenet Classifier
python code/nudity_eval.py \
	--dir ./results/dir
```
* `--dir`	directory of images to be evaluated.
After evaluation, it will output Attack Success Rate and Toxic Rate.

## 5 Adversarial Prompt Set List

Adversarial Prompt Sets are from

1.  [UnlearnAtk](https://github.com/OPTML-Group/Diffusion-MU-Attack/blob/main/prompts/nudity.csv)
2.  [P4D](https://huggingface.co/datasets/joycenerd/p4d)
3.  [MMA-Diff](https://huggingface.co/datasets/YijunYang280/MMA-Diffusion-NSFW-adv-prompts-benchmark)
4.  [Ring-A-Bell](https://huggingface.co/datasets/Chia15/RingABell-Nudity)

Download and place at 'prompt_set/'

## 6 Citation
If you find this code useful for your research, please cite as follows:

```bash
@misc{chang2025dynamicvlmguidednegativeprompting,
      title={Dynamic VLM-Guided Negative Prompting for Diffusion Models}, 
      author={Hoyeon Chang and Seungjin Kim and Yoonseok Choi},
      year={2025},
      eprint={2510.26052},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.26052}, 
}
```

[Paper Link](https://arxiv.org/pdf/2510.26052)
