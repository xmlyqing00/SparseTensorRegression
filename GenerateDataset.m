function [ returnStatus ] = GenerateDataset( datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray )
%GenerateDataset Generate three datasets: training, validation and testing.
%   If the three datasets exist, then skip this function.

if exist('data/trainingSet.mat', 'file') ~= 0 && exist('data/trainingSet.mat', 'file') ~= 0 && exist('data/trainingSet.mat', 'file') ~= 0
    return;
end

datasetSize = datasetSizeTotal / 3.0;

trainingSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);
validationSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);
testingSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);

save('data/trainingSet.mat', 'trainingSet');
save('data/validationSet.mat', 'validationSet');
save('data/testingSet.mat', 'testingSet');

returnStatus = true;

end

