## Dataset
The datasets are specifically described in "Data and code availability" section

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
## Reproduce
For example:
```bash
python main.py --data_type E15_5-S1
```
All data types are shown in params.py
