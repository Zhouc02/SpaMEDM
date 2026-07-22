## Dataset
The MISAR-seq mouse embryonic brain datasets and human lymph node D1 sample are obtained at https://zenodo.org/records/15681100. The human lymph node A1 sample, mouse spleen datasets, P22 mouse brain Spatial ATAC-RNA-seq and Spatial CUT&Tag-RNA-seq datasets are obtained at https://zenodo.org/records/10362607. The Spatial-CITE-seq human tonsil dataset is obtained at https://zenodo.org/records/13932144. The Slide-tags sequencing technology human melanoma single-cell RNA-ATAC dataset is obtained at https://singlecell.broadinstitute.org/single_cell/study/SCP2176/slide-tags-multiomic-snrna-seq-snatac-seq-on-human-melanoma#/. The mouse brain RNA-metabolomics dataset is obtained from https://drive.google.com/drive/folders/113Si1InBZl64LBvPSwkoMgFELIwljfVA?usp=drive_link.

## Requirement
### Conda envs
```bash
conda create -n SpaMEDM python=3.8.19
conda activate SpaMEDM
conda install pytorch==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install r-base=4.4.1
pip install torch-geometric==2.6.1
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install scanpy==1.9.1
pip install pandas==1.5.0
pip install numpy==1.22.3
conda install matplotlib=3.4.3
pip install --user scikit-misc
pip install leidenalg
pip install s-dbw
pip install rpy2==3.4.1
```
### R envs
```bash
install.packages("mclust")
```
## Run
On your own device, for example:
```bash
python main.py --data_type E15_5-S1
```
A specific [tutorial](./Supplementary/Tutorial/Tutorial.ipynb) is provided, which can reproduce most types of results.

All data types are shown in [params.py](./params.py), you need to set your own data path in this file and may modify the h5ad file name in [preprocess.py](./preprocess.py). 

We also provide our [model weights](./pth) and spatial domain identification [results](./results).
