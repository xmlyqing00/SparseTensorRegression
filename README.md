# Sparse Tensor Regression

This repository implements the core function of paper *"Sparse Multi-Response Tensor Regression for Alzheimer's Disease Study With Multivariate Clinical Assessments"*. Firstly, we generate the patterns and the datasets. Then we use Sparse Tensor Regression method to estimate our model. The training results are compared with the generated patterns.

## How to Use

1. Install Matlab [tensor_toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html).
2. Download this repository in the folder.
3. Run the script *main* in the Matlab environment.

## Experiments

- Small sample test: Passed.

  We choose a 3x3 matrix as pattern and train the model. The training process converges after 50 iterations.
  ![equation](http://latex.codecogs.com/gif.latex?Generated%20Pattern%3D%5Cbegin%7Bbmatrix%7D%200%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%201%5C%5C%200%20%26%201%20%26%200%20%5Cend%7Bbmatrix%7D)
  ![equation](http://latex.codecogs.com/gif.latex?Estimated%20Pattern%20%3D%20%5Cbegin%7Bbmatrix%7D%20-0.003%20%26%200.001%20%26%200.012%20%5C%5C%200.012%20%26%200.130%20%26%200.897%20%5C%5C%20-0.008%20%26%200.922%20%26%200.087%20%5C%5C%20%5Cend%7Bbmatrix%7D)

- Medium sample test: Undergoing.

## Documents

- **main.m** Main script contains three components: 1. Generate the patterns. 2. Generate the datasets. 3. Estimate the  models.

- **CalcObjFunc.m** Calculate the value of objective function (5).
- **ComposeTensor.m** Compose the components to a tensor. This is the inverse operation of CP decomposition.
- **DecomposeTensor.m** Decompose the tensor to the components by the input argument *rank*. The actual decomposition method is CP decomposition *cp_als*.
- **GenerateData.m** Generate the data by input arguments and store them in the *./data/* folder.
- **GenerateDataset.m** Generate all three datasets: training, validation and testing. It calls *GenerateData.m* to generate data.
- **GeneratePattern.m** Generate the default patterns and store them in the *./data/* folder.
- **InitModels.m** Initialize the models by random values.
- **LoadModels.m** Load the model from files. These files may be the snapshot of the last training, saved by *SaveTrainingStatus.m*.
- **SaveTrainingStatus.m** Save the snapshot of the training process, including temporal models values.
- **TrainModel.m** Estimate the model using Sparse Tensor Regression method  with the generated datasets.

## References

1. Li, Z., Suk, H. I., Shen, D., & Li, L. (2016). Sparse Multi-Response Tensor Regression for Alzheimer's Disease Study With Multivariate Clinical Assessments. *IEEE transactions on medical imaging*, *35*(8), 1927-1936.
2. Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox Version 2.6, Available online, February 2015. URL: http://www.sandia.gov/~tgkolda/TensorToolbox/. 





