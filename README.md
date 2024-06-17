## Security Assessment of Hierarchical Federated Deep Learning ##

This repository contains the code for our paper that has been accepted by [ICANN 2024](https://e-nns.org/icann2024/)
You can find the paper in this link: [here] ()

## Setup ##
Our environment has been tested on Nvidia V100 GPUs povided by [JADE](https://www.jade.ac.uk/).

To start Clone our repo and create a virtual environment
```
git clone https://github.com/dalqattan/HFL-attacks.git
cd HFL-attacks
python -m venv HFL-env
source HFL-env/bin/activate
pip install -r requirements.txt
```

## Experiment configurations ##

Create a TOML file to write the experiment configurations. 

You can find sample configuration files in the configuration folder.

You can create your own or simply choose one of the configuration files to reproduce our results.

## Run the experiment. ##
```
python HFL.py -config configuration/[ConfigurationFile].toml
```
Note: Replace [ConfigurationFile] with any of the file names in the configuration folder.

Each server's model will be stored in the location specified in the configuration file at every global aggregation round.
We suggest storing any experiment output in the output folder in this repo.

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
