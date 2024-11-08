# Application of KANs to quartet inference tasks

## Prerequisites

For all scripts Python x.x was used. The python packages used can be installed via
```sh
pip install (--user) -r packages_required.txt
```
if python3 and pip are already installed.


## Network for distinguishing Farris and Felsenstein trees

The training and test data for a network distinguishing alignments simulated under Farris and Felsenstein trees is saved in the folder data/processed/zone.

If it is not available the training data can be generated via 
```sh
./1_preprocess_zone_train_data.sh
```
and the test data via
```sh
./1_preprocess_zone_test_data.sh
```
in the folder data/preprocessing.


A train and test scripts for the network can be found within the scripts folder.

Running
```sh
python3 train.py <config>
```
a network is trained using the hyperparameters defined in the config-file (see e.g. config/config_F-zoneNN.yaml). The trained models are saved within the models folder.

An already trained network can be tested by executing:
```sh
python3 test.py -m <model>
```
The results will be saved in the results folder.
