## Security Assessment of Hierarchical Federated Deep Learning ##

This repository contains the code for our paper that has been accepted by [ICANN 2024](https://e-nns.org/icann2024/)
You can find the paper at this link: [here] ()

**Abstract:**

## Setup ##
Our environment has been tested on Ubuntu 22, CUDA 11.8 with A100, A800 and A6000.
Clone our repo and create conda environment
```
git clone https://github.com/buaacyw/MeshAnything.git && cd MeshAnything
conda create -n MeshAnything python==3.10.13
conda activate MeshAnything
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Experiment configurations ##

Create a TOML file to write the experiment configurations. 
You can find sample configuration files in the configuration folder.
You can create your own or simply choose one of the configuration files to reproduce our results.

## Run the experiment. ##
```
python path/to/FL.py -config path/to/[ConfigurationFile].toml
```
Note: Replace [ConfigurationFile] with any of the file names in the configuration folder.

Each server's model will be stored in the location specified in the configuration file at every global aggregation round.

## BibTeX ##
```
@inproceedings{?,
  title={Security Assessment of Hierarchical Federated Deep Learning},
  author={?},
  booktitle={International Conference on Artificial Neural Networks},
  pages={?--?},
  year={2024},
  organization={Springer}
}
```
