#!/bin/bash

# simulate data similar to strepsiptera data
python3 data_simulator_strepsiptera.py "../raw/zone/test/strepsiptera/quartet-tree-parameters/" "../raw/zone/test/strepsiptera/model-params-msa/param-table.tsv" 10 '../processed/zone/test/strepsiptera/sim/sim_freq_test.csv'
