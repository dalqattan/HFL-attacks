##Security Assessment of Hierarchical Federated Deep Learning'

This repository contains the code for our paper that has been accepted by [ICANN 2024](https://e-nns.org/icann2024/)
You can find the paper at this link: [here] ()

**Abstract:**

## Setup ##

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
