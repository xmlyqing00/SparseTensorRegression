function [ trainingSet, validationSet, testingSet ] = GenerateDataset( datasetSizeTotal, predictorSize, responseNum, correlation, noiseLevel, patternArray )
%GenerateDataset Generate three datasets: training, validation and testing.

datasetSize = datasetSizeTotal / 3.0;

trainingSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);
validationSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);
testingSet = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray);

end

