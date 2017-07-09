% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me

clc;
clear;
close all;

predictorSize = [64, 64];
GeneratePattern(predictorSize);

datasetSizeTotal = 750;
responseNum = 2;
correlation = 0.9;
noiseLevel = 5;
patternArray = [1, 2];
GenerateDataset(datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray);

lambda = 10;
rank = 2;
load('data/trainingSet.mat', 'trainingSet');
load('data/validationSet.mat', 'validationSet');
load('data/testingSet.mat', 'testingSet');
models = TrainModel(lambda, rank, trainingSet, validationSet);
