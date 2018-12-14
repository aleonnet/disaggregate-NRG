# Energy Disaggregation

Samir Sen, Fred Lu

We experiment with various approaches to disaggregate whole home energy signal 
into individual usage by appliance. 

Here we use the REDD dataset as written by Kolter et. al. 2009. This dataset 
contains meter data in both high and low settings over 6 homes in Boston, MA. 
We also experiment with data augmentation using weather metadata over the 
timesteps collected from NOAA. 

Under Disaggregate foler are our models:
  - Sparse Coding
  - Random Forest
  - SVM
  - Ridge Regression
  - Multi-task Elastic Net
  - Neural Network
  - KNN
  
