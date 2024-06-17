## Security Assessment of Hierarchical Federated Deep Learning ##

This repository contains the code for our paper that has been accepted by [ICANN 2024](https://e-nns.org/icann2024/)
You can find the paper at this link: [here] ()

**Abstract:**

## Setup ##
Our environment has been tested on Ubuntu 22, CUDA 11.8 with A100, A800 and A6000.
Clone our repo and create conda environment
```
git clone https://github.com/dalqattan/SecHFL.git
cd SecHFL
python -m venv SecHFL_env
source SecHFL_env/bin/activate
pip install -r requirements.txt
```

## Experiment configurations ##

Create a TOML file to write the experiment configurations. 
You can find sample configuration files in the configuration folder.
You can create your own or simply choose one of the configuration files to reproduce our results.

## Run the experiment. ##
```
python path/to/SecHFL/FL.py -config path/to/SecHFL/configuration/[ConfigurationFile].toml
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
