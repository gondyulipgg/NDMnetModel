# NDMnetModel
This repository includes the codes for the manuscript entitled "Numerical dispersion mitigation neural network with velocity model correction", which was submitted to Computers & Geosciences for peer review. A neural network-based method for suppressing numerical errors in seismograms: numerical dispersion and velocity model errors. 

# Code introduction
- [NDMnet.py](NDMnet.py): NDM-net model architecture 
- [dataset.py](dataset.py): code for converting a dataset into a suitable format for a neural network.
- [TrainingNDM-net.ipynb](TrainingNDM-net.ipynb): train the main program, where you can change hyperparameters and path. 
- [ResultsAnalysis.ipynb](ResultsAnalysis.ipynb): applying a neural network to test data and calculating the relative error.
- To test the algorithm, we used the Marmousi2 velocity model, which can be found with the [link](https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz), and to form a training set, we modeled seismograms on various grids and model discretizations. 


# Requirements
- cuda 8.0
- Python 3.9.18
- conda 23.3.1
- pytorch 1.12.1
- matplotlib 3.7.0
- numpy 1.23.1
- natsorted 8.2.0
