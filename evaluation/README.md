# Semi-supervised VOS evaluation

This package is derived from  <a href="https://davischallenge.org/davis2017/code.html" target="_blank">DAVIS 2017</a> evaluation implementation and used to evaluate semi-supervised video multi-object segmentation models for the <a href="https://www.vostdataset.org" target="_blank">VOST</a> dataset. 

## Installation
Download the code:
```bash
git clone https://github.com/TRI-ML/VOST.git
```
Install the required dependencies:
```bash
pip install numpy Pillow opencv-python pandas scikit-image scikit-learn tqdm scipy
```

## Evaluation
In order to evaluate your method on the validation set of VOST, execute the following command:
```bash
python evaluation_method.py --results_path PATH_TO_YOUR_RESULTS --dataset_path PATH_TO_VOST --set val
```

If you don't want to specify the dataset path every time, you can modify the default value in the variable `default_dataset_path` in `evaluation_method.py`. 

Once the evaluation has finished, two different CSV files will be generated inside the folder with the results: 
- `global_results-SUBSET.csv` contains the overall results for a certain `SUBSET`. 
- `per-sequence_results-SUBSET.csv` contain the per sequence results for a certain `SUBSET`.

If a folder that contains the previous files is evaluated again, the results will be read from the CSV files instead of recomputing them.

## Citation

Please cite the following papers in your publications if this code helps your research.

```latex
@inproceedings{tokmakov2023breaking,
  title={Breaking the “Object” in Video Object Segmentation},
  author={Tokmakov, Pavel and Li, Jie and Gaidon, Adrien},
  booktitle={CVPR},
  year={2023}
}
```

```latex
@article{Caelles_arXiv_2019,
  author = {Sergi Caelles and Jordi Pont-Tuset and Federico Perazzi and Alberto Montes and Kevis-Kokitsi Maninis and Luc {Van Gool}},
  title = {The 2019 DAVIS Challenge on VOS: Unsupervised Multi-Object Segmentation},
  journal = {arXiv},
  year = {2019}
}
```

```latex
@article{Pont-Tuset_arXiv_2017,
  author = {Jordi Pont-Tuset and Federico Perazzi and Sergi Caelles and Pablo Arbel\'aez and Alexander Sorkine-Hornung and Luc {Van Gool}},
  title = {The 2017 DAVIS Challenge on Video Object Segmentation},
  journal = {arXiv:1704.00675},
  year = {2017}
}
```

