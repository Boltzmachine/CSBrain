# [NeurIPS2025 Spotlight] CSBrain: Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding 🧠⚡

## Overview 🌟
CSBrain is a cutting-edge **Cross-scale Spatiotemporal Brain Foundation Model** designed for EEG decoding. By incorporating innovative techniques like **Cross-scale Spatiotemporal Tokenization (CST)** and **Structured Sparse Attention (SSA)**, CSBrain captures multi-scale dependencies in EEG signals. This enables better generalization across a wide range of EEG decoding tasks.

The model has shown superior performance compared to task-specific and foundation models across various EEG tasks, such as **emotion recognition**, **motor imagery**, and **seizure detection**.

## Key Features ✨
- **Cross-scale Spatiotemporal Tokenization (CST)**: Aggregates features within localized temporal windows and anatomical brain regions into compact, scale-aware token representations 🕒🌐.
- **Structured Sparse Attention (SSA)**: Models long-range dependencies across temporal windows and brain regions, while minimizing spurious dependencies 🔄🔍.
- **Generalizable Model**: Outperforms task-specific models across multiple EEG tasks without needing fine-tuning 🔧💪.
- **Masked Autoencoding Pretraining**: Learns generalizable representations from unlabeled EEG signals, enhancing transferability 🔑.

## Installation 🛠️

### Clone the Repository
Clone the CSBrain repository to your local machine:
```bash
git clone https://github.com/yuchen2199/CSBrain.git
cd CSBrain
```

### Install Dependencies
Please follow the environment installation and data preprocessing steps outlined in [CBraMod](https://github.com/wjq-learning/CBraMod).  
In addition to the benchmarks tested in CBraMod, we have also included widely used open datasets such as Siena, HMC, and TUSL for evaluation. The data preprocessing scripts for these new datasets can be found in the `preprocessing` folder.

### Pretraining CSBrain from Scratch
Once the pretraining dataset is processed, you can pretrain CSBrain using the following script:
```bash
bash sh/pretrain_CSBrain.sh
```

### Finetuning CSBrain on Downstream Datasets
We have provided fine-tuning scripts for 16 datasets used in this work in the `sh` folder. After completing data preprocessing, you can run the fine-tuning scripts directly, for example:
```bash
bash sh/finetune_CSBrain_BCIC.sh
```
You can also download the pre-trained weights of CSBrain and the weights for downstream tasks we provided from [Google Drive]([link](https://drive.google.com/drive/folders/1-GsVVewRM0B93H08yts5m53yU2whxYvj?usp=sharing)).

### References 📚
If you find our paper/code useful, please consider citing our work:
```bash
@inproceedings{
zhou2025csbrain,
title={CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding},
author={Yuchen Zhou and Jiamin Wu and Zichen Ren and Zhouheng Yao and Weiheng Lu and Kunyu Peng and Qihao Zheng and Chunfeng Song and Wanli Ouyang and Chao Gou},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=agcXjEHmyW}
}
```

### Acknowledgments
We sincerely thank the previous works and open-source efforts, including LaBraM ([GitHub](https://github.com/935963004/LaBraM)), BIOT ([GitHub](https://github.com/ycq091044/BIOT)), EEGPT ([GitHub](https://github.com/BINE022/EEGPT)), and CBraMod ([GitHub](https://github.com/wjq-learning/CBraMod)), for their invaluable contributions.


