% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me

clc;
clear;
close all;

predictorSize = [64, 64];
GeneratePattern(predictorSize);

datasetSizeTotal = 450;
responseNum = 3;
correlation = 0.9;
noiseLevel = 10;
patternArray = [1, 2];
GenerateDataset(datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray);

lambda = 100;
load('trainingSet.mat', 'trainingSet');
load('validationSet.mat', 'validationSet');
load('testingSet.mat', 'testingSet');
models = TrainModel(lambda, trainingSet, validationSet);
