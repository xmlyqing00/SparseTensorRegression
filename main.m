% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me

clc;
clear all;
close all;

predictorSize = [64, 64];
GeneratePattern(predictorSize);

datasetSizeTotal = 750;
responseNum = 3;
correlation = 0.9;
noiseLevel = 10;
patternArray = [1, 2];
GenerateDataset(datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray);

lambda = 10;
models = TrainModel(lambda, trainingSet, validationSet);
