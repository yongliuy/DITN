<p align="center">
  <img src="./assets/ditn_logo.png" height=35>
</p>

# Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution
[Yong Liu](https://scholar.google.com/citations?user=DT0LPIEAAAAJ&hl=en&oi=sra), 
[Hang Dong](https://scholar.google.com/citations?user=4DKepr8AAAAJ&hl=en&oi=sra), 
[Boyang Liang](https://scholar.google.com/citations?user=6wsdO2oAAAAJ&hl=en&oi=sra), 
[Songwei Liu](https://scholar.google.com/citations?user=crxAeIEAAAAJ&hl=en&oi=sra), 
Qingji Dong, 
Kai Chen, 
Fangmin Chen, 
Lean Fu, 
[Fei Wang](http://www.aiar.xjtu.edu.cn/info/1046/1242.htm)<br/>

[Paper](https://dl.acm.org/doi/10.1145/3581783.3612128) | [arXiv](https://arxiv.org/abs/2308.02794) | [Poster](./assets/poster.png) | [BibTeX](#bibtex) 


:sparkling_heart: If our DITN is helpful to your researches or projects, please help star this repository. Thanks! :hugs: 

![visitors](https://visitor-badge.laobi.icu/badge?page_id=yongliuy/DITN) 
<img alt="GitHub" src="https://img.shields.io/badge/license-Apache_2.0-brightgreen">
[![GitHub Stars](https://img.shields.io/github/stars/yongliuy/DITN?style=social)](https://github.com/yongliuy/DITN/)


>Recent years have witnessed a few attempts of vision transformers for single image super-resolution (SISR). 
Since the high resolution of intermediate features in SISR models increases memory and computational requirements, efficient SISR transformers are more favored. 
Based on some popular transformer backbone, many methods have explored reasonable schemes to reduce the computational complexity of the self-attention module while achieving impressive performance. 
However, these methods only focus on the performance on the training platform (e.g., Pytorch/Tensorflow) without further optimization for the deployment platform (e.g., TensorRT). 
Therefore, they inevitably contain some redundant operators, posing challenges for subsequent deployment in real-world applications. 
In this paper, we propose a deployment-friendly transformer unit, namely UFONE (i.e., UnFolding ONce is Enough), to alleviate these problems. 
In each UFONE, we introduce an Inner-patch Transformer Layer (ITL) to efficiently reconstruct the local structural information from patches and a Spatial-Aware Layer (SAL) to exploit the long-range dependencies between patches. 
Based on UFONE, we propose a Deployment-friendly Inner-patch Transformer Network (DITN) for the SISR task, which can achieve favorable performance with low latency and memory usage on both training and deployment platforms. 
Furthermore, to further boost the deployment efficiency of the proposed DITN on TensorRT, we also provide an efficient substitution for layer normalization and propose a fusion optimization strategy for specific operators. 
Extensive experiments show that our models can achieve competitive results in terms of qualitative and quantitative performance with high deployment efficiency.
<p align="center">
<img src=assets/method.png width="1000px"/>
</p>


## Update
- **2023.07.06**: Create this repository.

## TODO
- [ ] New project website
- [ ] The training scripts
- [ ] The model deployment guide
- [x] ~~Releasing pretrained models~~
- [x] ~~The inference scripts~~




## Requirements
```
conda create -n ditn python=3.8
conda activate ditn
pip3 install -r requirements.txt
```

## Applications
### :snowboarder: Demo on Base Evaluation Dataset
[<img src="assets/imgsli_1.jpg" height="280px"/>](https://imgsli.com/MTk1NjE2) [<img src="assets/imgsli_2.jpg" height="280px"/>](https://imgsli.com/MTk1NjIz)
[<img src="assets/imgsli_3.jpg" height="260px"/>](https://imgsli.com/MTk1NjI1) [<img src="assets/imgsli_4.jpg" height="260px">](https://imgsli.com/MTk1NjI2)


### :whale: Demo on Real-world Image SR
[<img src="assets/imgsli_5.jpg" height="266px"/>](https://imgsli.com/MTk1NjQ4)
[<img src="assets/imgsli_6.jpg" height="266px"/>](https://imgsli.com/MTk1NjUw) [<img src="assets/imgsli_7.jpg" height="265px">](https://imgsli.com/MTk1NjUx)
[<img src="assets/imgsli_8.jpg" height="265px"/>](https://imgsli.com/MTk1NjUy) [<img src="assets/imgsli_9.jpg" height="265px">](https://imgsli.com/MTk1NjUz)

## Pretrained Models
- Download the DITN pretrained models from [Google Drive](https://drive.google.com/drive/folders/1XpHW27H5j2S4IH8t4lccgrgHkIjqrS-X?usp=drive_link).


## Running Examples

- Prepare your test images and run the ``DITN/test.py`` with cuda on command line: 
### :rocket: Bicubic Image Super-resolution
```bash
DITN/$CUDA_VISIBLE_DEVICES=<GPU_ID> python test.py --scale [2|3|4] --indir [the path of LR images] --outdir [the path of HR results] --model_path [the path of the pretrained model]/DITN_[ |Tiny|Real]_[x2|x3|x4].pth
```

### :trophy: Real-world Image Super-resolution
```bash
DITN/$CUDA_VISIBLE_DEVICES=<GPU_ID> python test.py --scale [2|3|4] --indir [the path of LR images] --outdir [the path of HR results] --model_path [the path of the pretrained model]/DITN_Real_GAN_[x2|x3|x4].pth
```

## How to Deployment in Realistic Scenarios
- Coming soon...


## Acknowledgement
This work was supported in part by the National Key Research and Development Program of China under Grant 2022YFB3303800, in part by National Major Science and Technology Projects of China under Grant 2019ZX01008101.


## License
This project is released under the [Apache 2.0 license](./LICENSE). Redistribution and use should follow this license.


## BibTeX
If you find this project useful for your research, please use the following BibTeX entry.
```
@inproceedings{liu2023unfolding,
  title={Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution},
  author={Liu, Yong and Dong, Hang and Liang, Boyang and Liu, Songwei and Dong, Qingji and Chen, Kai and Chen, Fangmin and Fu, Lean and Wang, Fei},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7952--7960},
  year={2023}
}
```


