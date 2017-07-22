function [ returnStatus ] = GenerateDataset( datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray, override )
%GenerateDataset Generate three datasets: training, validation and testing.
%   Parameters:
%       datasetSizeTotal: The total number of samples among all datasets.
%       predictorSize: The size of the predictor X.
%       responseNum: The number of responses.
%       correlation: The correlation among each response of noise item.
%       noiseLevel: The variance of normal distribution of noise item.
%       override: The boolean flag decides whether to re-create new datasets.
%
%   If the override flag is false and the three datasets exist, then skip 
%   this function.
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

if override == false && ...
    exist('data/trainingSet.mat', 'file') ~= 0 && ...
    exist('data/trainingSet.mat', 'file') ~= 0 && ...
    exist('data/trainingSet.mat', 'file') ~= 0

    disp('Generate the datasets. Skip.');
    return;
end

datasetSize = datasetSizeTotal / 3.0;

trainingSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);
validationSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);
testingSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);

save('data/trainingSet.mat', 'trainingSet');
save('data/validationSet.mat', 'validationSet');
save('data/testingSet.mat', 'testingSet');

disp('Generate the datasets. Finish.');
returnStatus = true;

end

