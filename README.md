# MeGA: Hybrid Mesh-Gaussian Head Avatar for High-Fidelity Rendering and Head Editing
The official repo for "[MeGA: Hybrid Mesh-Gaussian Head Avatar for High-Fidelity Rendering and Head Editing](insert)"

<p align="center">
<a href="https://arxiv.org/abs/2404.19026"><img src="https://img.shields.io/badge/Arxiv-2404.19026-B31B1B.svg"></a>
<a href="https://conallwang.github.io/MeGA_Pages/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
</p>

## :mega: Updates

[28/4/2024] The official repo is initialized.

## Abstract

Creating high-fidelity head avatars from multi-view videos is a core issue for many AR/VR applications. However, existing methods usually struggle to obtain high-quality renderings for all different head components simultaneously since they use one single representation to model components with drastically different characteristics (e.g., skin vs. hair). In this paper, we propose a Hybrid Mesh-Gaussian Head Avatar (MeGA) that models different head components with more suitable representations. Specifically, we select an enhanced FLAME mesh as our facial representation and predict a UV displacement map to provide per-vertex offsets for improved personalized geometric details. To achieve photorealistic renderings, we obtain facial colors using deferred neural rendering and disentangle neural textures into three meaningful parts. For hair modeling, we first build a static canonical hair using 3D Gaussian Splatting. A rigid transformation and an MLP-based deformation field are further applied to handle complex dynamic expressions. Combined with our occlusion-aware blending, MeGA generates higher-fidelity renderings for the whole head and naturally supports more downstream tasks. Experiments on the NeRSemble dataset demonstrate the effectiveness of our designs, outperforming previous state-of-the-art methods and supporting various editing functionalities, including hairstyle alteration and texture editing.

## Pipeline

<p align="center">
<img src="assets/pipeline_git.png" width="800"/>
</p>

## TODO

- [x] Release the project page
- [ ] Add more results to the project page
- [ ] Release the code. 

## Citation

If you find this code useful for your research, please consider citing:
```

```

