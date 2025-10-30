<h2 align="center">RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models</h2>
<p align="center">
    <a href="https://github.com/RT-DETRs/RT-DETRv4/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/RT-DETRs/RT-DETRv4">
    </a>
    <a href="https://github.com/RT-DETRs/RT-DETRv4/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/RT-DETRs/RT-DETRv4">
    </a>
    <a href="https://github.com/RT-DETRs/RT-DETRv4/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/RT-DETRs/RT-DETRv4?color=pink">
    </a>
    <a href="https://github.com/RT-DETRs/RT-DETRv4">
        <img alt="stars" src="https://img.shields.io/github/stars/RT-DETRs/RT-DETRv4">
    </a>
    <a href="https://arxiv.org/abs/2510.25257">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2510.25257-red">
    </a>
    <a href="mailto:zjliao25@stu.pku.edu.cn">
        <img alt="email" src="https://img.shields.io/badge/contact-email-yellow">
    </a>
</p>

---

This is the official implementation of the paper:
* [RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models](https://arxiv.org/abs/2510.25257)

## 🚀 Overview

**RT-DETRv4 is the new version of the state-of-the-art real-time object detector family, RT-DETR.** It introduces a cost-effective and adaptable distillation framework that leverages the powerful representations of Vision Foundation Models (VFMs) to enhance lightweight detectors.


## 📣 News
* **[2025.10.30]** Repo created, and code will be open-sourced very soon!

## ⚡ Performance

RT-DETRv4 achieves new state-of-the-art results on the COCO dataset, outperforming previous real-time detectors.

| Model | AP (val) | Latency (T4) | FPS (T4) |
| :--- | :---: | :---: | :---: |
| RT-DETRv4-S | 49.7 | 3.66 ms | 273 |
| RT-DETRv4-M | 53.5 | 5.91 ms | 169 |
| RT-DETRv4-L | 55.4 | 8.07 ms | 124 |
| RT-DETRv4-X | 57.0 | 12.90 ms | 78 |

## 🤖 Model Zoo

*(Links to pretrained RT-DETRv4-S/M/L/X models will be provided here.)*

## 🛠️ Getting Started

*(Installation instructions, training, and evaluation commands will be added here.)*

## Citation

If you find this work helpful, please consider citing:
```bibtex
@article{liao2025rtdetrv4,
  title={RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models},
  author={Liao, Zijun and Zhao, Yian and Shan, Xin and Yan, Yu and Liu, Chang and Lu, Lei and Ji, Xiangyang and Chen, Jie},
  journal={arXiv preprint arXiv:2510.25257},
  year={2025}
}
