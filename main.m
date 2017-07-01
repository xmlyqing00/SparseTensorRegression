% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me

clc;
clear all;
close all;

GeneratePattern([64, 64]);

datasetSizeTotal = 750;
responseNum = 3;
correlation = 0.9;
noiseLevel = 10;
patternArray = [1, 2];
predictorSize = [64, 64];
[trainingSet, validationSet, testingSet] = GenerateDataset(datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray);

