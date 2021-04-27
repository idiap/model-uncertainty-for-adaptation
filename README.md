# Code for paper [*Uncertainty Reduction for Model Adaptation in Semantic Segmentation*](http://publications.idiap.ch/downloads/papers/2021/Sivaprasad_CVPR_2021.pdf) at CVPR 2021

In this package, we provide our PyTorch code for out CVPR 2021 paper on Model Adaptation for Segmentation. If you use our code, please cite us:

```bibtex
@inproceedings{teja2021uncertainty,
  author = {S, Prabhu Teja. and Fleuret, F.},
  title = {Uncertainty Reduction for Model Adaptation in Semantic Segmentation},
  booktitle = {Proceedings of the IEEE international conference on Computer Vision and Pattern Recognition (CVPR)},
  note = {(to appear)},
  year = {2021}
}
```

The PDF version of the paper is available [here](http://publications.idiap.ch/downloads/papers/2021/Sivaprasad_CVPR_2021.pdf).

## Requirements

We use PyTorch for the experiments. The conda environment required to run these codes can be installed by

`conda create --name ucr --file spec-file.txt`

While we aren't aware of any python version specific idiosyncracies, we tested this on Python 3.7 on Debian, with the above `spec-file.txt`. If you find any missing details, or have trouble getting it to run, please create an issue.

## Training the network

### Downloading pre-trained models

We use the pretrained models provided by [MaxSquareLoss](https://github.com/ZJULearning/MaxSquareLoss) at [https://drive.google.com/file/d/1wLffQRljXK1xoqRY64INvb2lk2ur5fEL/view](https://drive.google.com/file/d/1wLffQRljXK1xoqRY64INvb2lk2ur5fEL/view) into a folder named `pretrained`

### Setting up paths

First, the paths to the Cityscapes dataset has to be set in `datasets/new_datasets.py` in the dataset's constructor. The path to NTHU cities dataset can be set in `utils/argparser.py` in line 15 at `DATA_TGT_DIRECTORY` or can be added to the command line call at with `--data-tgt-dir`. The code trains the network and evaluates its performance and writes it into the log file in the `savedir` called `training_logger`.

### Running the code

Then code can be run with

```python
python do_segm.py --city {city} --no-src-data --freeze-classifier --unc-noise --lambda-ce 1 --lambda-ent 1  --save {savedir} --lambda-ssl 0.1
```

where `city` can in `Rome` or `Rio` or `Tokyo` or `Taipei`, and `savedir` is the path to save the logs and models. 

## Acknowledgements

This code borrows parts from [MaxSquareLoss](https://github.com/ZJULearning/MaxSquareLoss) (the network definitions, and pretrained models) and [CRST](https://github.com/yzou2/CRST) (class balanced pseudo-label generation). The author thanks Evann Courdier for parts of the clean `datasets` code.

## License

This software is distributed with the **MIT** license which pretty much means that you can use it however you want and for whatever reason you want. All the information regarding support, copyright and the license can be found in the [`LICENSE`](./LICENSE) file in the repository.
