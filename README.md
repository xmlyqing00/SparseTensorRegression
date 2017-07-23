# Sparse Tensor Regression

This repository implements the core function of paper *"Sparse Multi-Response Tensor Regression for Alzheimer's Disease Study With Multivariate Clinical Assessments"*. Firstly, we generate the patterns and the datasets. Then we use Sparse Tensor Regression method to estimate our model. The training results are compared with the generated patterns.

## How to Use

1. Install Matlab [tensor_toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html).
2. Download the release [package](https://github.com/LyqSpace/SparseTensorRegression/releases) and unzip it in the folder.
3. Run the script *demo_reg* in the Matlab environment.

## Introduction

### Multivariate Tensor Regression Model
The multivariate tensor regression model is Eq.(3)

![equation](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BY%7D%20%3D%20%5Cmathbf%7BB%7D%20%5Cmathbf%7BX%7D%20&plus;%20%5Cmathbf%7Be%7D)

![equation](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BY%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%3C%5Csum_%7Br%3D1%7D%5E%7BR%7D%5Cmathbf%7B%5Cbeta_%7B11%7D%5E%7B%28r%29%7D%7D%5Ccirc%5Ccdots%5Ccirc%5Cmathbf%7B%5Cbeta_%7B1D%7D%5E%7B%28r%29%7D%7D%2C%5Cmathbf%7BX%7D%3E%20%5C%5C%20%5Cvdots%5C%5C%20%3C%5Csum_%7Br%3D1%7D%5E%7BR%7D%5Cmathbf%7B%5Cbeta_%7Bq1%7D%5E%7B%28r%29%7D%7D%5Ccirc%5Ccdots%5Ccirc%5Cmathbf%7B%5Cbeta_%7BqD%7D%5E%7B%28r%29%7D%7D%2C%5Cmathbf%7BX%7D%3E%20%5Cend%7Bbmatrix%7D%20&plus;%20%5Cmathbf%7Be%7D)

### Datasets
Firstly, we generate the model **B** following specified patterns and simulate **X** and **e** following  a normal distribution. The response variables **Y** are calcualted by **B**, **X** and **e**. Given *n* independent and identically distributed samples 

![equation](http://latex.codecogs.com/gif.latex?%5C%7B%28%5Cmathbf%7BX_1%7D%2C%5Cmathbf%7BY_1%7D%29%2C%28%5Cmathbf%7BX_2%7D%2C%5Cmathbf%7BY_2%7D%29%2C%5Ccdots%2C%28%5Cmathbf%7BX_n%7D%2C%5Cmathbf%7BY_n%7D%29%2C%5C%7D)

following chapter IV. We sperate them into three datasets: training set, validation set and testing set. 

### Objective Function
Then we minimize the objective function Eq.(5) using mini-batch gradient descending algorithm to estimate the models' parameters.

![equation](http://latex.codecogs.com/gif.latex?%5Cell%28%5Cmathbf%7BB%7D_1%2C%5Ccdots%2C%5Cmathbf%7BB%7D_q%29%20%3D%20%5Cmathbf%7BL%7D%28%5Cmathbf%7BB%7D_1%2C%5Ccdots%2C%5Cmathbf%7BB%7D_q%29%20&plus;%20%5Clambda%5Cmathbf%7BJ%7D%28%5Cmathbf%7BB%7D_1%2C%5Ccdots%2C%5Cmathbf%7BB%7D_q%29)

where

![equation](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BL%7D%28%5Cmathbf%7BB%7D_1%2C%5Ccdots%2C%5Cmathbf%7BB%7D_q%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bq%7D%5CBig%28Y_%7Bij%7D-%3C%5Csum_%7Br%3D1%7D%5ER%5Cbeta_%7Bj1%7D%5E%7B%28r%29%7D%5Ccirc%5Ccdots%5Ccirc%5Cbeta_%7BjD%7D%5E%7B%28r%29%7D%2C%5Cmathbf%7BX%7D_i%3E%5CBig%29%5E2)

![equation](http://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Baligned%7D%5Cmathbf%7BJ%7D%28%5Cmathbf%7BB%7D_1%2C%5Ccdots%2C%5Cmathbf%7BB%7D_q%29%20%26%3D%20%5Csum_%7Bd%3D1%7D%5ED%5Csum_%7Br%3D1%7D%5ER%5Csum_%7Bk%3D1%7D%5E%7Bp_D%7D%5CBig%28%5Csum_%7Bj%3D1%7D%5Eq%7B%5Cbeta_%7Bjdk%7D%5E%7B%28r%29%7D%7D%5E2%5CBig%29%5E%7B1/2%7D%20%5C%5C%20%26%3D%20%5Cfrac%7B%5Clambda%7D%7B2%7D%20%5Csum_%7Bd%3D1%7D%5ED%5Csum_%7Br%3D1%7D%5ER%5Csum_%7Bk%3D1%7D%5E%7Bp_D%7D%5CBig%28%5Cmu_%7Bdk%7D%5E%7B%28r%29%7D%7C%7C%5Cmathbf%7Bb%7D%7C%7C%5E2&plus;%5Cfrac%7B1%7D%7B%5Cmu_%7Bdk%7D%5E%7B%28r%29%7D%7D%5CBig%29%5Cend%7Baligned%7D)

### Minimize Objective Function

Here, we have two ways to find the minimum of the objective function. One is naive mini-batch gradient descending (TrainModelGradDesc.m), the other one is updating the models by closed form solution (TrainModelDerivative.m), proposed in the paper. The first one is simple and obvious, but its performance is inferior to the second one. We use the second method and alternatively update the parameters as below.

![equation](http://latex.codecogs.com/gif.latex?%5Cmu_%7Bdk%7D%5E%7B%28r%29%7D%20%3D%20%5Cfrac%7B1%7D%7B%7C%7C%5Cmathbf%7Bb_%7Bdk%7D%5E%7B%28r%29%7D%7D%7C%7C%7D)

![equation](http://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20vec%28%5Cmathbf%7BB_%7Bjd%7D%7D%29%20%3D%20%5CBig%28%5Csum_%7Bi%3D1%7D%5En%5Cwidetilde%7B%5Cmathbf%7BX%7D%7D_%7Bijd%7D%5Cwidetilde%7B%5Cmathbf%7BX%7D%7D_%7Bijd%7D%5E%7B%5Cmathbf%7BT%7D%7D&plus;%5Cfrac%7B%5Clambda%7D%7B2%7Ddiag%5Cbig%28%5Cmu_%7Bd1%7D%5E%7B%281%29%7D%2C%5Ccdots%2C%5Cmu_%7Bdp_d%7D%5E%7B%281%29%7D%2C%5Ccdots%2C%5Cmu_%7Bd1%7D%5E%7B%28R%29%7D%2C%5Ccdots%2C%5Cmu_%7Bdp_d%7D%5E%7B%28R%29%7D%5Cbig%29%5CBig%29%5E%7B-1%7D%5Csum_%7Bi%3D1%7D%5En%5Cmathbf%7BY%7D_%7Bij%7D%5Cwidetilde%7B%5Cmathbf%7BX%7D%7D_%7Bijd%7D)

### Experiments

The experiment results are below. We compare two training methods: one is naive mini-batch gradient descending (TrainModelGradDesc.m), the other one is updating the model by closed form solution (TrainModelDerivative.m), proposed in the paper. Obviously, the first training method is time-consuming and its results are less accurate than the second training method.

Mini-batch gradient descending method needs about 24 hours to reach convergence, while derivative closed form updating method needs about 5 minutes to reach convergence.

![summary](https://github.com/LyqSpace/SparseTensorRegression/blob/master/summary.png)

## Documents

- **demo_reg.m** demo script contains three components: 1. Generate the patterns. 2. Generate the datasets. 3. Estimate the  models.
- **CalcObjFunc.m** Calculate the value of objective function Eq.(5).
- **ComposeTensor.m** Compose the components to a tensor. This is the inverse operation of CP decomposition.
- **DecomposeTensor.m** Decompose the tensor to the components by the input argument *rank*. The actual decomposition method is CP decomposition *cp_als*.
- **DrawTrainingResults.m** Draw the training results that were saved in the *training/* folder and compare between the generated pattern and the estimated patterns.
- **GenerateData.m** Generate the data by input arguments and store them in the *./data/* folder.
- **GenerateDataset.m** Generate all three datasets: training, validation and testing. It calls *GenerateData.m* to generate data.
- **GeneratePattern.m** Generate the default patterns and store them in the *./data/* folder.
- **InitModels.m** Initialize the models by random values.
- **LoadModels.m** Load the model from files. These files may be the snapshot of the last training, saved by *SaveTrainingStatus.m*.
- **SaveTrainingStatus.m** Save the snapshot of the training process, including temporal models values.
- **TrainModelDerivative.m** We let the partial derivative to be zero and update the model B by its closed form solution.
- **TrainModelGradDesc.m** Estimate the model by mini-batch gradient descending.

## License

- Author: Space Liang
- License: Apache License 2.0

## References

1. Li, Z., Suk, H. I., Shen, D., & Li, L. (2016). Sparse Multi-Response Tensor Regression for Alzheimer's Disease Study With Multivariate Clinical Assessments. *IEEE transactions on medical imaging*, *35*(8), 1927-1936.
2. Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox Version 2.6, Available online, February 2015. URL: http://www.sandia.gov/~tgkolda/TensorToolbox/. 





