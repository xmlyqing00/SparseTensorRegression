% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me

clc;
clear;
close all;

predictorSize = [64, 64];
GeneratePattern(predictorSize);

datasetSizeTotal = 1500;
responseNum = 3;
correlation = 0.9;
noiseLevel = 10;
patternArray = [1, 2, 3];
GenerateDataset(datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray);

lambda = 10;
rank = 3;
load('data/trainingSet.mat', 'trainingSet');
load('data/validationSet.mat', 'validationSet');
load('data/testingSet.mat', 'testingSet');
models = TrainModel(lambda, rank, trainingSet, validationSet);
