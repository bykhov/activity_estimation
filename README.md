This is the supplementary material for the paper "Deep Learning Based Method for Activity Estimation from Short-Duration Gamma Spectroscopy Recordings" by D. Bykhovsky and T. Trigano.
The paper is currently under review and the supplementary material is provided for the reviewers' convenience.

This repository contains the two main parts of the project:
1. The code for the extraction of the signal parameters from the gamma spectroscopy measurement signal. The details of the code are provided in the [signal_analysis_example](/signal_analysis_example) directory in the documentation [file](/signal_analysis_example/README.md).
2. The code for the training of the deep learning model for the activity estimation. The corresponding code is provided in the [gamma_sym_performance](/gamma_sym_performance) directory:
   * [gamma_simulator.py](/gamma_sym_performance/gamma_simulator.py) - the script for the simulation of the gamma spectroscopy measurements.
   * [create_dataset.py](/gamma_sym_performance/create_dataset.py) - the script for the creation of the dataset from the simulated gamma spectroscopy measurements.
   * [train_model.py](/gamma_sym_performance/train_model.py) - the script for the training of the deep learning model.
   * [test_model_exp.ipynb](/gamma_sym_performance/test_model_exp.ipynb) - the Jupyter notebook for the testing of the trained model on the experimental data into the [signal.mat](/gamma_sym_performance/signal.mat) file.