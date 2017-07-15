% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me

clc;
clear;
close all;

predictorSize = [64, 64];
GeneratePattern(predictorSize);

datasetSizeTotal = 750;
responseNum = 1;
correlation = 0.9;
noiseLevel = 10;
patternArray = [4];
predictorSize = [3, 3];
GenerateDataset(datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray);

lambda = 1;
rank = 1;
load('data/trainingSet.mat', 'trainingSet');
load('data/validationSet.mat', 'validationSet');
load('data/testingSet.mat', 'testingSet');
models = TrainModelDerivation(lambda, rank, trainingSet, validationSet);
