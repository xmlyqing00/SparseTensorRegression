% Author:   Space Liang
% Homepage: www.lyq.me
% Email:    root [at] lyq.me
%
% More details can be found at this repo homepage: 
%   https://github.com/LyqSpace/SparseTensorRegression
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

clc;
clear;
close all;

disp('Sparse Tensor Regression');
disp('Copyright 2017, Space Liang. Email: root [at] lyq.me');
disp(' ');
disp('More details can be found at this repo homepage:');
disp('  https://github.com/LyqSpace/SparseTensorRegression');
disp(' ');

% Generate the patterns as the standards for models.
predictorSize = [64, 64];
override = false;
GeneratePattern(predictorSize, override);

% Generate the three datasets.
datasetSizeTotal = 1500;
responseNum = 4;
correlation = 0.9;
noiseLevel = 10;
patternArray = [1, 2, 3, 4];
override = false;
GenerateDataset(datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray, override);

% Train the model.
lambda = 10;
rank = 3;
load('data/trainingSet.mat', 'trainingSet');
load('data/validationSet.mat', 'validationSet');
load('data/testingSet.mat', 'testingSet');
% The best training method.
models = TrainModelDerivation(lambda, rank, trainingSet, validationSet);
% This training method is slow and inacurate.
% models = TrainModelGradDesc(lambda, rank, trainingSet, validationSet);

% Draw the training results.
DrawTrainingResults(responseNum);
