# [NeurIPS 2024] Activating Self-Attention for Multi-Scene Absolute Pose Regression
This is the official pytorch implementation of [Activating Self-Attention for Multi-Scene Absolute Pose Regression](https://arxiv.org/abs/2411.01443).

Authors: [Miso Lee](https://leemiso.notion.site/), [Jihwan Kim](https://www.linkedin.com/in/damien1224/), [Jae-Pil Heo](https://sites.google.com/site/jaepilheo)

![Motivation](./assets/motivation.png)


## Requirements
- Python 3.8.0
- Pytorch 1.10.1+cu111
- CUDA 11.1
- 1 RTX Titan


## Installation
```bash
conda create -n actmst python==3.8
conda activate actmst
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


## Downloads
- Datasets:
    [CambridgeLandmarks](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3#dataset) / [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
- Checkpoints:
    [Google Drive](https://drive.google.com/drive/folders/1R1yQ701O_6CAf7RcfsywUpNFmUvadDwZ?usp=sharing)


## Training
```bash
python main.py \
    --model_name ems-transposenet \
    --mode train \
    --backbone_path ./models/backbones/efficient-net-b0.pth \
    --dataset_path ${DATASET_PATH} \                        # Dataset directory path
    --scene all \
    --labels_file ./datasets/${DATASET}/all_scenes.csv \    # Path to labels file for all scenes
    --config_file ${CONFIG}.json \                          # Configuration file
    --experiment ${EXP_NAME} \                              # Experiment name
    --gpu ${GPU_NUM}                                        # GPU index
```
For Cambridge Landmarks, it is required to change ```config_file``` to ```CambridgeLandmarks_config.json``` for initial training and ```CambridgeLandmarks_finetune_config.json``` for fine-tuning (see details in [multi-scene-pose-transformer](https://github.com/yolish/multi-scene-pose-transformer)). 


## Evaluation
```bash
python main.py \
    --model_name ems-transposenet \
    --mode test \
    --backbone_path ./models/backbones/efficient-net-b0.pth \
    --dataset_path ${DATASET_PATH} \                        # Dataset directory path
    --scene ${SCENE} \                                      # Scene to be evaluated
    --labels_file ./datasets/${DATASET}/${SCENE}_test.csv \ # Path to labels file for the test scene
    --config_file ${CONFIG}.json \                          # Configuration file
    --checkpoint_path ${CKPT_SAVE_PATH} \                   # Checkpoint file path
    --experiment ${EXP_NAME} \                              # Experiment name
    --gpu ${GPU_NUM}                                        # GPU index
```

## Citation
If our work is useful, please consider the following citation:
```
@inproceedings{NEURIPS2024_43d7bc00,
 author = {Lee, Miso and Kim, Jihwan and Heo, Jae-Pil},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {38508--38529},
 publisher = {Curran Associates, Inc.},
 title = {Activating Self-Attention for Multi-Scene Absolute Pose Regression},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/43d7bc009cf5171e7af77a91ee4bb890-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```


## Acknowledgement
This repository is built based on [multi-scene-pose-transformer](https://github.com/yolish/multi-scene-pose-transformer) repository.
Thank you for the great work.


## License
This project is released under the MIT license.
See [LICENSE](LICENSE) for additional details.