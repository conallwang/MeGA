# MeGA: Hybrid Mesh-Gaussian Head Avatar for High-Fidelity Rendering and Head Editing
The official repo for "[MeGA: Hybrid Mesh-Gaussian Head Avatar for High-Fidelity Rendering and Head Editing](https://arxiv.org/abs/2404.19026)"

<p align="center">
<a href="https://arxiv.org/abs/2404.19026"><img src="https://img.shields.io/badge/Arxiv-2404.19026-B31B1B.svg"></a>
<a href="https://conallwang.github.io/MeGA_Pages/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
</p>

<p align="center">
  <img src="./assets/git_demo.gif" />
</p>

## :mega: Updates


[07/12/2024] All data of other subjects are released [here](https://drive.google.com/drive/folders/1N7pzrTtwBKQ033SZG5kukpFvT2RqoLXg?usp=sharing). Thanks for [ZiXuan](https://scholar.google.com/citations?user=3i9GwyIAAAAJ) providing the cloud storage. 

[07/8/2024] The data and pretrained models of Subject 306 have been released [here](https://drive.google.com/drive/folders/1R7fNJnWu6ZSqbIvpUWbAUfb5qdq2a8sp?usp=sharing)!

[01/8/2024] The Codes has been released!

[06/5/2024] Add more results to the project page.

[28/4/2024] The official repo is initialized.

## TODO

- [x] Release the project page
- [x] Add more results to the project page
- [x] Release the codes
- [x] Release the data and Subject 306's pretrained model.
- [x] Upload the data of Subject 218, 304.
- [x] Upload all data of other subjects.
- [ ] Update the codes to the latest version.

## Pipeline

![pipeline_git](https://github.com/user-attachments/assets/461d5f5a-5451-407d-928c-5310478e855d)


## Setup

### Environment

Here, we provide commands that are needed to build the [conda](https://docs.anaconda.com/miniconda/#latest-miniconda-installer-links) environment:
```shell
# 1. create a new conda env & activate
conda create -n mega python=3.9
conda activate mega

# 2. run our scripts to install requirements
./create_env.sh
```

### Data & Pretrained models

We use the same 9 subjects from NeRSemble dataset as GaussianAvatars in our experiments. Based on their provided data, we additionally generate depth maps and face parsing results. All pre-processed data and models that are used to reproduce the results of Subject 306 are provided [here](https://drive.google.com/drive/folders/1R7fNJnWu6ZSqbIvpUWbAUfb5qdq2a8sp?usp=sharing). 

> Whether you want to train or test our methods, you need to download the data and decompress it into somewhere, e.g., /path/to/nersemble

For more subjects' data, please download from [here](https://drive.google.com/drive/folders/1N7pzrTtwBKQ033SZG5kukpFvT2RqoLXg?usp=sharing). 

### Training

To train a full MeGA avatar (taking Subject 306 as an example), you need to take two steps.

First, train a canonical hair model using
```shell
# Before execute the following commands, you need to change every path ('/path/to/...') to your specific path.
# Including files: ['./scripts/train_hair.sh', './configs/nersemble/306/hair.yaml']

cd /path/to/MeGA
bash ./scripts/train_hair.sh
```

After that, your hair model will be saved in your specified directory (i.e., $WORKSPACE/$VERSION/checkpoint_reset.pth).

Next, train the full avatar model using
```shell
# Also changing every path ('/path/to/...') to your specific path.
# Including files: ['./scripts/train_full.sh', './configs/nersemble/306/full.yaml']

cd /path/to/MeGA
bash ./scripts/train_full.sh
```

### Testing (Including computing metrics)

If you want to only render images in the test dataset and valid dataset or compute metrics, you can run
```shell
cd /path/to/MeGA
bash ./scripts/metrics.sh
```

The script will render images first and then compute metrics automaticly.


### Funny editting

As mentioned in our paper, MeGA supports some human head editing. All related codes are in [./funny_demo](./funny_demo/).

#### Hair alteration

To perform hair alteration (e.g., alternate Subject 218's hair to 306's hair), you can run

```shell
cd /path/to/MeGA
bash ./scripts/alter_hair.sh
```

#### Texture editting

We have provided some 2d painting images in the preprocessed data (/path/to/nersemble/preprocess/306/306_EMO-1_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/images/00000_08_*.png). 

You can also produce your own 2d painting images and put them to the 3d head avatar with our scripts.

```shell
cd /path/to/MeGA
bash ./scripts/paint.sh
```

This process will take some time (several minutes) to optimize.

### Render videos using pre-trained models

We take the painted avatar above as an example. The painted avatar will be saved in somewhere like '/path/to/checkpoints/MeGA/0801/train_306_b16_MeGA/duola', and you can further render sequences using painted avatars:
```shell
cd /path/to/MeGA
bash ./scripts/render.sh
```

The results will be saved in somewhere like '/path/to/checkpoints/MeGA/0801/train_306_b16_MeGA/duola/exp3_eval'. If you want a video result, please execute './scripts/img2video.sh' (using ffmpeg).
```shell
cd /path/to/MeGA
bash ./scripts/img2video.sh /path/to/checkpoints/MeGA/0801/train_306_b16_MeGA/duola/exp3_eval/renders
```

The video can be generated in '/path/to/checkpoints/MeGA/0801/train_306_b16_MeGA/duola/exp3_eval/output.mp4'.

## Citation

If you find this code useful for your research, please consider citing:
```
@article{wang2024mega,
  title={MeGA: Hybrid Mesh-Gaussian Head Avatar for High-Fidelity Rendering and Head Editing},
  author={Wang, Cong and Kang, Di and Sun, He-Yi and Qian, Shen-Han and Wang, Zi-Xuan and Bao, Linchao and Zhang, Song-Hai},
  journal={arXiv preprint arXiv:2404.19026},
  year={2024}
}
```
