% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me

clc;
clear;
close all;

predictorSize = [64, 64];
override = false;
GeneratePattern(predictorSize, override);

datasetSizeTotal = 1500;
responseNum = 4;
correlation = 0.9;
noiseLevel = 10;
patternArray = [1, 2, 3, 4];
override = false;
GenerateDataset(datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray, override);

lambda = 10;
rank = 3;
load('data/trainingSet.mat', 'trainingSet');
load('data/validationSet.mat', 'validationSet');
load('data/testingSet.mat', 'testingSet');
models = TrainModelDerivation(lambda, rank, trainingSet, validationSet);
