function [ returnStatus ] = GenerateDataset( datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray )
%GenerateDataset Generate three datasets: training, validation and testing.
%   If the three datasets exist, then skip this function.

if exist('trainingSet.mat', 'file') ~= 0 && exist('trainingSet.mat', 'file') ~= 0 && exist('trainingSet.mat', 'file') ~= 0
    return;
end

datasetSize = datasetSizeTotal / 3.0;

trainingSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);
validationSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);
testingSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);

save('trainingSet.mat', 'trainingSet');
save('validationSet.mat', 'validationSet');
save('testingSet.mat', 'testingSet');

returnStatus = true;

end

