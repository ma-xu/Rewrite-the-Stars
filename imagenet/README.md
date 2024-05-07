# [Rewrite the Stars](https://arxiv.org/abs/2403.19967) - CVPR'24


## Image Classification for DemoNet and StarNet
### 1. Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; timm==0.6.13; einops; fvcore; h5py;

### 2. Dataset
data prepare: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```



### 3. Train DemoNet and StarNet
We show how to train models on 8 GPUs. We retrain all models based on the relased file, and achieve slightly better results than reported.

```bash
# Train DemoNet variants
python3 -m torch.distributed.launch --nproc_per_node=8 train_imagenet.py --data {path-to-imagenet} --model {demonet-variants} -b 256 --lr 4e-3 --model-ema
# Train StarNet variants (--drop-path 0. for S1, 0.01 for S2,S3, and 0.02 for S4)
# Based on previous works, efficient / small networks don't need strong augmentations.
python3 -m torch.distributed.launch --nproc_per_node=8 train_imagenet.py --data {path-to-imagenet} --model {starnet-variants} -b 256 --lr 3e-3 --weight-decay 0.05 --aa rand-m1-mstd0.5-inc1 --cutmix 0.2 --color-jitter 0. --drop-path 0.
```

### 4. Pretrained checkpoints

| Model      | Params | FLOPs | Top-1 | Mobile / GPU / CPU latency (ms) | Download      |
|------------|--------|-------|-------|---------------------------------|---------------|
| starnet_s1 | 2.9M   | 425M  | 73.6  | 0.7 / 2.3 / 4.3                 | [[model](https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar)] [[log](https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.csv)] |
| starnet_s2 | 3.7M   | 547M  | 74.7  | 0.7 / 2.0 / 4.5                 | [[model](https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar)] [[log](https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.csv)] |
| starnet_s3 | 5.8M  | 757M  | 77.4  | 0.9 / 2.7 / 6.7                 | [[model](https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar)] [[log](https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.csv)] |
| starnet_s4 | 7.5M   | 1075M | 78.8  | 1.0 / 3.7 / 9.4                 | [[model](https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar)] [[log](https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.csv)] |



## Benchmark ONNX speed on CPU and GPU :v::v::v:
We also provide a script to help benchmark model latency on different platforms, which is important but always not available. 

In this script, we can benchmark **different models**, **different input resolution**, **different hardwares** (ONNX on CPU, ONNX on GPU, Pytorch on GPU) using [ONNX Runtime](https://github.com/microsoft/onnxruntime).

Meanwhile, we can **save a detailed log file** ({args.results_file}, e.g., debug.csv) that can log almost all detailed information (including model related logs, data related, benchmark results, system / hardware related logs) for each benchmark.

### 1. Requirements

onnxruntime-gpu==1.13.1; onnx==1.13.0; tensorrt==8.5.2.2; torch>=1.7.0; torchvision>=0.8.0; timm==0.6.13; fvcore; thop; py-cpuinfo; 

### 2. Run benchmark script
```bash
# Please feel free to add / modify configs if necessary
# Benchmark results will be printed and saved to {args.results_file}, appending to the last row. 
CUDA_VISIBLE_DEVICES=0 python3 benchmark_onnx.py --model {model-name} --input-size 3 244 244 --benchmark_cpu
```

## Benchmark CoreML speed on iPhone

### 1. Requirements
coremltools;

### 2. Convert Pytorch models to CoreML models
```bash
# Convert pytorch mdoels to CoreML models, optional load checkpoint
python3 export_coreml.py --model {model-name}
```

### 3. Benchmark CoreML speed
Please check the CoreML benchmark tool here: [MobileOne Benchmark](https://github.com/apple/ml-mobileone/tree/main/ModelBench).

We appreciate the open-sourced CoreML benchmark tool from MobileOne group.

