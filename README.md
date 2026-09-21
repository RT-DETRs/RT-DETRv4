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

<p align="center">
  <strong>🎉 We are excited to announce that RT-DETRv4 has been accepted to ECCV 2026!</strong>
</p>

---

This is the official implementation of the paper:
* [RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models](https://arxiv.org/abs/2510.25257)


## 🚀 Overview

**RT-DETRv4 is the new version of the state-of-the-art real-time object detector family, RT-DETR.** It introduces a cost-effective and adaptable distillation framework that leverages the powerful representations of Vision Foundation Models (VFMs) to enhance lightweight detectors.
<p align="center">
  <img src="./figures/rtv4_ap_latency.png" alt="Image" width="49%">
</p>
<details open>
<summary> 🎥 Demo: Furious 7 </summary>

We challenge **RT-DETRv4** with the high-speed scene from *Furious 7*. Even under extreme motion blur, rapid camera pans, and chaotic lighting, the model maintains consistent tracking and high recall, proving its reliability in dynamic environments.

<div align="center">

https://github.com/user-attachments/assets/e610b13b-a6bb-420b-ac6f-a752f61d0d84

</div>

</details>

<details open>
<summary> 🚁 Demo: VisDrone (aerial imagery) </summary>

**RT-DETRv4-L** fine-tuned on VisDrone at 960×960 with DINOv3 distillation, running on two VisDrone2019-VID val sequences: night (left) and day (right). Dense, tiny targets (pedestrians, cars, vans, trucks, tricycles) seen from a drone. Fine-tuned S/M/L weights and full training/evaluation commands are in [Usage → VisDrone](#2-usage).

<p align="center">
  <img src="./figures/visdrone_demo.gif" alt="RT-DETRv4-L detections on VisDrone aerial footage, night (left) and day (right)" width="90%">
</p>

<sub>4 s excerpt at 30 fps; here is the [full-length video](https://huggingface.co/dronefreak/visdrone-rtdetrv4-l/resolve/main/assets/demo_banner.mp4). Detections at confidence ≥ 0.35. For illustration only; not part of the reported metrics.</sub>

</details>

## ⚡ Performance

RT-DETRv4 achieves new state-of-the-art results on the COCO dataset, outperforming previous real-time detectors.

| Model | AP | AP50 | AP75 | Latency (T4) | FPS (T4) | Config | Log | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: |:---------------------------------------------:|:----------------------------:|:---:|
| RT-DETRv4-S | 49.8 | 67.1 | 54.0 | 3.66 ms | 273 | [yml](./configs/rtv4/rtv4_hgnetv2_s_coco.yml) | [log](./logs/RTv4-S-hgnet.log) | [ckpt](https://drive.google.com/file/d/1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z) |
| RT-DETRv4-M | 53.7 | 71.0 | 58.4 | 5.91 ms | 169 | [yml](./configs/rtv4/rtv4_hgnetv2_m_coco.yml) | [log](./logs/RTv4-M-hgnet.log) | [ckpt](https://drive.google.com/file/d/1O-YpP4X-quuOXbi96y2TKkztbjroP5mX) |
| RT-DETRv4-L | 55.4 | 73.0 | 60.3 | 8.07 ms | 124 | [yml](./configs/rtv4/rtv4_hgnetv2_l_coco.yml) | [log](./logs/RTv4-L-hgnet.log) | [ckpt](https://drive.google.com/file/d/1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8) |
| RT-DETRv4-X | 57.0 | 74.6 | 62.1 | 12.90 ms | 78 | [yml](./configs/rtv4/rtv4_hgnetv2_x_coco.yml) | [log](./logs/RTv4-X-hgnet.log) | [ckpt](https://drive.google.com/file/d/19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3) |
## 📣 News

  * **[2026.06.18]** 🎉 RT-DETRv4 has been accepted to **ECCV 2026**! We sincerely thank all contributors and the community for your support!
  * **[2025.11.17]** Code, configs and checkpoints fully released! Thanks for your attention, and feel free to ask any questions!
  * **[2025.10.30]** Repo created, and code will be open-sourced very soon!

## Content

  * [1. Getting Started](#1-getting-started)
  * [2. Usage](#2-usage)
  * [3. Tools](#3-tools)
  * [4. Citation](#4-citation)
  * [5. Acknowledgement](#5-acknowledgement)

-----

> This repository also supports the reproduction of [DEIM](./configs/deim/), [D-FINE](./configs/dfine/), and [RT-DETRv2](./configs/rtv2/). Simply run the corresponding configuration files.

## 1\. Getting Started

### Setup

```shell
conda create -n rtv4 python=3.11.9
conda activate rtv4
pip install -r requirements.txt
```

### Data Preparation

<details>
<summary> COCO2017 Dataset </summary>

1.  Download COCO2017 from [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) or [COCO](https://cocodataset.org/#download).
2.  Modify paths in [coco\_detection.yml](./configs/dataset/coco_detection.yml)

```yaml
train_dataloader:
    img_folder: /data/COCO2017/train2017/
    ann_file: /data/COCO2017/annotations/instances_train2017.json
val_dataloader:
    img_folder: /data/COCO2017/val2017/
    ann_file: /data/COCO2017/annotations/instances_val2017.json
```

</details>

<details>
<summary>Custom Dataset</summary>

To train on your custom dataset, you need to organize it in the COCO format. Follow the steps below to prepare your dataset:

1.  **Set `remap_mscoco_category` to `False`:**

    This prevents the automatic remapping of category IDs to match the MSCOCO categories.

    ```yaml
    remap_mscoco_category: False
    ```

2.  **Organize Images:**

    Structure your dataset directories as follows:

    ```shell
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── ...
    ```

      * **`images/train/`**: Contains all training images.
      * **`images/val/`**: Contains all validation images.
      * **`annotations/`**: Contains COCO-formatted annotation files.

3.  **Convert Annotations to COCO Format:**

    If your annotations are not already in COCO format, you'll need to convert them. You can use the following Python script as a reference or utilize existing tools:

    ```python
    import json

    def convert_to_coco(input_annotations, output_annotations):
        # Implement conversion logic here
        pass

    if __name__ == "__main__":
        convert_to_coco('path/to/your_annotations.json', 'dataset/annotations/instances_train.json')
    ```

4.  **Update Configuration Files:**

    Modify your [custom\_detection.yml](./configs/dataset/custom_detection.yml).

    ```yaml
    task: detection

    evaluator:
      type: CocoEvaluator
      iou_types: ['bbox', ]

    num_classes: 777 # your dataset classes
    remap_mscoco_category: False

    train_dataloader:
      type: DataLoader
      dataset:
        type: CocoDetection
        img_folder: /data/yourdataset/train
        ann_file: /data/yourdataset/train/train.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~
      shuffle: True
      num_workers: 4
      drop_last: True
      collate_fn:
        type: BatchImageCollateFunction

    val_dataloader:
      type: DataLoader
      dataset:
        type: CocoDetection
        img_folder: /data/yourdataset/val
        ann_file: /data/yourdataset/val/ann.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~
      shuffle: False
      num_workers: 4
      drop_last: False
      collate_fn:
        type: BatchImageCollateFunction
    ```

</details>

### Teacher Model Preparation

Our framework uses a pre-trained Vision Foundation Model (VFM) as the teacher. We use the **ViT-B/16-LVD-1689M** model from DINOv3.

  * **Repository:** [DINOv3](https://github.com/facebookresearch/dinov3)
  * **Weights:** [Downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)


### Configuring DINOv3 Teacher

Specify the paths to your local DINOv3 repository and the downloaded checkpoint in the model's configuration file `./configs/rtv4/rtv4_hgnetv2_${model}_coco.yml` and find the `teacher_model` section:

```yaml
teacher_model:
  type: "DINOv3TeacherModel"
  dinov3_repo_path: dinov3/
  dinov3_weights_path: pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
```

Update the `dinov3_repo_path` and `dinov3_weights_path` to match your local setup.

## 2\. Usage

<details open>
<summary> COCO2017 </summary>

1.  Training

    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/rtv4/rtv4_hgnetv2_${model}_coco.yml --use-amp --seed=0
    ```

2.  Testing

    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/rtv4/rtv4_hgnetv2_${model}_coco.yml --test-only -r model.pth
    ```

3.  Tuning

    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/rtv4/rtv4_hgnetv2_${model}_coco.yml --use-amp --seed=0 -t model.pth
    ```

</details>

<details>
<summary> VisDrone (aerial imagery, 960×960) </summary>

RT-DETRv4 S/M/L fine-tuned from the COCO checkpoints on VisDrone2019-DET, with DINOv3 distillation at 960×960. Weights, configs and full model cards are on Hugging Face.

| Model | AP | AP50 | Params | FLOPs (960²) | Config | Weights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| RT-DETRv4-S | 27.4 | 46.3 | 10.4M | 52.5B | [yml](./configs/rtv4/rtv4_hgnetv2_s_visdrone_960.yml) | [🤗 HF](https://huggingface.co/dronefreak/visdrone-rtdetrv4-s) |
| RT-DETRv4-M | 27.8 | 46.9 | 19.4M | 122.4B | [yml](./configs/rtv4/rtv4_hgnetv2_m_visdrone_960.yml) | [🤗 HF](https://huggingface.co/dronefreak/visdrone-rtdetrv4-m) |
| RT-DETRv4-L | 28.9 | 48.3 | 30.9M | 197.5B | [yml](./configs/rtv4/rtv4_hgnetv2_l_visdrone_960.yml) | [🤗 HF](https://huggingface.co/dronefreak/visdrone-rtdetrv4-l) |

<sub>COCO-style AP on the VisDrone2019-DET **test** split (1610 images), EMA weights, ≤100 detections per image; the `valid` split is used for checkpoint selection. VisDrone is released for non-commercial research (CC BY-NC-SA 3.0). The models are distilled from DINOv3, whose license is included in each Hugging Face repo.</sub>

**Configs.** `rtv4_hgnetv2_${model}_visdrone_960.yml` (recommended, used for the released weights) is 960×960 with DINOv3 distillation; it includes `rtv4_hgnetv2_${model}_visdrone.yml`, the 640×640 baseline without distillation. `${model}` is `s`, `m` or `l` (`x` configs are provided, but there is no released X checkpoint).

**Data.** The configs expect a COCO-format VisDrone export: `train/`, `valid/` and `test/`, each with the images and an `_annotations.coco.json`, 11 classes with ids 0-10 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor, others). The `img_folder` / `ann_file` paths in `configs/rtv4/rtv4_hgnetv2_${model}_visdrone.yml` are the original author's machine paths; edit them for your setup.

**DINOv3 teacher.** The `_960` configs distill from DINOv3, so set up the teacher first (see [Teacher Model Preparation](#teacher-model-preparation)). It is needed for training and, since the solver builds it from the config, also for `--test-only`. Inference and ONNX export with the released weights do not need it.

1.  Training (fine-tune from the COCO checkpoint of the same size, `${MODEL}` is `S`, `M` or `L`)

    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port=7777 --nproc_per_node=3 train.py -c configs/rtv4/rtv4_hgnetv2_${model}_visdrone_960.yml --use-amp --seed=0 -t RTv4-${MODEL}-hgnet.pth
    ```

2.  Testing (COCO metrics on the test split; the config's `val_dataloader` points at `valid`, so override it)

    ```shell
    CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=1 train.py -c configs/rtv4/rtv4_hgnetv2_${model}_visdrone_960.yml --test-only -r outputs/rtv4_hgnetv2_${model}_visdrone_960/best_stg2.pth -u val_dataloader.dataset.img_folder=/path/to/VisDrone/COCO/test val_dataloader.dataset.ann_file=/path/to/VisDrone/COCO/test/_annotations.coco.json
    ```

3.  Evaluation report (per-class AP, precision/recall, PR/F1/confusion plots, params/FLOPs, demo images/video, and a Hugging Face model card for each model)

    ```shell
    pip install -r tools/model_card/requirements.txt
    python tools/model_card/generate_model_card.py --manifest tools/model_card/manifests/visdrone.yml
    ```

    Edit the `eval` paths (and optional `banner` clips) in the manifest first. Output goes to `model_cards/` (git-ignored).

4.  Inference with the released weights

    ```shell
    hf download dronefreak/visdrone-rtdetrv4-${model} --local-dir weights/visdrone-${model}
    python tools/inference/torch_inf.py -c weights/visdrone-${model}/config.yaml -r weights/visdrone-${model}/model.pth --input image.jpg --device cuda:0
    ```

</details>

<details>
<summary> Customizing Batch Size </summary>

For example, if you want to double the total batch size when training RT-DETRv4-L on COCO2017, here are the steps you should follow:

1.  **Modify your [dataloader.yml](./configs/base/dataloader.yml)** to increase the `total_batch_size`:

    ```yaml
    train_dataloader:
        total_batch_size: 64  # Previously it was 32, now doubled
    ```

2.  **Modify your [rtv4\_hgnetv2\_l\_coco.yml](./configs/rtv4/rtv4_hgnetv2_l_coco.yml)**. Here’s how the key parameters should be adjusted:

    ```yaml
    optimizer:
      type: AdamW
      params:
        -
          params: '^(?=.*backbone)(?!.*norm|bn).*$'
          lr: 0.000025  # doubled, linear scaling law
        -
          params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
          weight_decay: 0.

    lr: 0.0005  # doubled, linear scaling law
    betas: [0.9, 0.999]
    weight_decay: 0.0001  # need a grid search

    ema:  # added EMA settings
        decay: 0.9998  # adjusted by 1 - (1 - decay) * 2
        warmups: 500  # halved

    lr_warmup_scheduler:
        warmup_duration: 250  # halved
    ```

</details>

<details>
<summary> Customizing Input Size </summary>

If you'd like to train **RT-DETRv4** on COCO2017 with an input size of 320x320, follow these steps:

1.  **Modify your [dataloader.yml](./configs/base/dataloader.yml)**:

    ```yaml
    train_dataloader:
      dataset:
          transforms:
              ops:
                  - {type: Resize, size: [320, 320], }
      collate_fn:
          base_size: 320

    val_dataloader:
      dataset:
          transforms:
              ops:
                  - {type: Resize, size: [320, 320], }
    ```

2.  **Modify your [rtv4\_base.yml](./base/rtv4_base.yml)** (or the relevant base config file):

    ```yaml
    eval_spatial_size: [320, 320]
    ```

</details>

## 3\. Tools

<details>
<summary> Deployment </summary>

1.  Setup

    ```shell
    pip install onnx onnxsim onnxscript
    ```

2.  Export onnx

    ```shell
    python tools/deployment/export_onnx.py --check -c configs/rtv4/rtv4_hgnetv2_${model}_coco.yml -r model.pth
    ```

3.  Export [tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

    ```shell
    trtexec --onnx="model.onnx" --saveEngine="model.engine"
    ```

    > **Note:** On TensorRT 10.x/11.x, `trtexec` builds strongly-typed networks by
    > default and no longer accepts the `--fp16` builder flag (`Unknown option: --fp16`).
    > Precision now follows the types declared in the ONNX graph itself; drop the flag
    > as shown above. If you're on an older TensorRT release that still uses the
    > weakly-typed builder, `--fp16` is still valid there.

</details>

<details>
<summary> Inference (Visualization) </summary>

1.  Setup

    ```shell
    pip install -r tools/inference/requirements.txt
    ```

2.  Inference (onnxruntime / tensorrt / torch)

    Inference on images and videos is now supported.

    ```shell
    python tools/inference/onnx_inf.py --onnx model.onnx --input image.jpg  # or video.mp4
    python tools/inference/trt_inf.py --trt model.engine --input image.jpg
    python tools/inference/torch_inf.py -c configs/rtv4/rtv4_hgnetv2_${model}_coco.yml -r model.pth --input image.jpg --device cuda:0
    ```

</details>

<details>
<summary> Benchmark </summary>

1.  Setup

    ```shell
    pip install -r tools/benchmark/requirements.txt
    ```

2.  Model FLOPs, MACs, and Params

    ```shell
    python tools/benchmark/get_info.py -c configs/rtv4/rtv4_hgnetv2_${model}_coco.yml
    ```

3.  TensorRT Latency

    ```shell
    python tools/benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
    ```

</details>

<details>
<summary> Fiftyone Visualization </summary>

1.  Setup

    ```shell
    pip install fiftyone
    ```

2.  Voxel51 Fiftyone Visualization ([fiftyone](https://github.com/voxel51/fiftyone))

    ```shell
    python tools/visualization/fiftyone_vis.py -c configs/rtv4/rtv4_hgnetv2_${model}_coco.yml -r model.pth
    ```

</details>

<details>
<summary> Others </summary>

1.  Auto Resume Training

    ```shell
    bash tools/reference/safe_training.sh
    ```

2.  Converting Model Weights

    ```shell
    python tools/reference/convert_weight.py model.pth
    ```

</details>

## 4\. Citation

If you find this work helpful, please consider citing:

```bibtex
@article{liao2025rtdetrv4,
  title={RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models},
  author={Zijun Liao and Yian Zhao and Xin Shan and Yu Yan and Chang Liu and Lei Lu and Xiangyang Ji and Jie Chen},
  journal={arXiv preprint arXiv:2510.25257},
  year={2025}
}
```

## 5\. Acknowledgement

Our work is built upon [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [D-FINE](https://github.com/Peterande/D-FINE), [DEIM](https://github.com/Intellindust-AI-Lab/DEIM) and Teacher Model [DINOv3](https://github.com/facebookresearch/dinov3).
Thanks to these remarkable works!
