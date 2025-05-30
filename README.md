# Multi-view Consistent 3D Panoptic Scene Understanding

Xianzhu Liu<sup>1</sup>, 
Xin Sun<sup>1</sup>, 
[Haozhe Xie](https://haozhexie.com/about)<sup>2</sup>, 
Zonglin Li<sup>1</sup>, 
[Ru Li](https://liru0126.github.io)<sup>1</sup>, 
[Shengping Zhang](https://spzhang.wordpress.com)<sup>1</sup>

<sup>1</sup>Harbin Institute of Technology, Weihai, China&nbsp;&nbsp;
<sup>2</sup>Nanyang Technological University, Singapore

![Overview](https://github.com/aipixel/MVC-PSU/blob/main/images/Overview.png?raw=true)

## Changelog

- [2024/12/19] The repo is created.
- [2025/5/29] The test code has been released.

## Installation 📥

Moreover, this repository introduces an integrated 3D Panoptic Scene Understanding Benchmark implemented in Python 3.8, PyTorch 1.12 and CUDA 11.3. 

1. You can use the following command to install PyTorch with CUDA 11.3. 
```
conda create -n ssc python=3.8
conda activate ssc
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

2. Install dependencies:
```
pip install -r requirements.txt

```

## Datasets and Pretrained Models 🛢️
For the datasets used in this paper, please refer to the download and preprocessing instructions provided in [Panoptic-Lifting](https://github.com/nihalsid/panoptic-lifting?tab=readme-ov-file)

You can download our pre-trained checkpoints [here](https://drive.google.com/drive/folders/1q9wZZOiOFv-erM4-1w8PInz3UvKyLN_2)


## Inference and Evaluation 🚩

We provide an example to use our code.
1. Please download the pretrained checkpoints and unzip.

2. Use the `render_panopli.py` script to render. Example: 
``` 
python inference/render_panopli.py pretrained_ckpts/hypersim001008/checkpoints/hypersim001008.ckpt True
```
This will render the outputs to ``` runs/<experiment> ``` folder. 

3. Use the `evaluate.py` script for calculating metrics. Example:
``` 
python inference/evaluate.py --root_path ./data/hypersim/hypersim001008 --exp_path runs/<experiment>
```


## Training 👩🏽‍💻
This repository only contains the inference code for MVC-PSU. The training code will be released in our subsequent work.

## Cite this work

```
@inproceedings{liu2025multi,
  title={Multi-view Consistent 3D Panoptic Scene Understanding},
  author={Liu, Xianzhu and Sun, Xin and Xie, Haozhe and Li, Zonglin and Li, Ru and Zhang, Shengping},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={5613--5621},
  year={2025}
}
```