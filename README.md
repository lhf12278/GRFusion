The full code will be uploaded when we get it organized!

#  GRFusion
The code of "Generation and Reconmbination for Multifocus Image Fusion with Free Number of Inputs".

- [*[ArXiv]*](https://arxiv.org/abs/2309.04657)

---

![Abstract](assets/overview.png)

---


## To Train
The training code for both the focus detection and fusion parts is provided here.

If you want to train the focus detection network : Run "**CUDA_VISIBLE_DEVICES=0 python train_fd.py**".

If you want to train the fusion network : Run "**CUDA_VISIBLE_DEVICES=0 python train_fusion.py**".


## To Test
The test code for both the focus detection and fusion parts is also provided here.

If you want to test the focus detection network : Run "**CUDA_VISIBLE_DEVICES=0 python test_fd.py**".

If you want to test the fusion network : Run "**CUDA_VISIBLE_DEVICES=0 python test_fusion.py**".


## Recommended Environment
We strongly recommend that you use Conda as a package manager.

```shell
# create virtual environment
conda create -n GRFusion python=3.10
conda activate GRFusion
# select pytorch version yourself
# install SegMiF requirements
pip install -r requirements.txt
```

## Test datasets download
Lytro, MFI-WHU, and MFFW Dataset can be downloaded in 
- [*[Lytro]*](https://mansournejati.ece.iut.ac.ir/content/lytro-multi-focus-dataset)
- [*[MFI-WHU]*](https://github.com/HaoZhang1018/MFI-WHU)
- [*[MFFW]*](https://www.semanticscholar.org/paper/MFFW%3A-A-new-dataset-for-multi-focus-image-fusion-Xu-Wei/4c0658f338849284ee4251a69b3c323908e62b45)

## Our results
![Abstract](assets/results3.png)

## Citation

If this work has been helpful to you, please feel free to cite our paper!



