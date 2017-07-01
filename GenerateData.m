function [ dataset ] = GenerateData( datasetSize, predictorSize, responseNum, correlation, noiseLevel, patternArray )
%GenerateData Generate dataset by specified arguments.
%   datasetSize: the size of the dataset.
%   predictorSize: the size of the predictor X.
%   responseNum: the number of responses.
%   correlation: the correlation among each response noise item.
%   noiseLevel: the variance of normal distribution of noise item.
%   
%   Formula: Y = BX + e
%   Y is the response and it calculated by B, X and e.
%   B is the model and it takes the value of 0 or 1 following specified patterns.
%   X is the observation and it follows a normal distribution.
%   e is the noise item and it follows a normal distribution with mean zero
%   and specified covariance.

patterns = cell(1, length(patternArray));
for i = 1:length(patternArray)
    patternIndex = patternArray(i);
    patternFileName = strcat('pattern', num2str(patternIndex), '.mat');
    loadStruct = load(patternFileName);
    patterns{i} = loadStruct.pattern;
end

models = cell(1, responseNum);
for i = 1:responseNum
    models{i} = patterns{mod(i, length(patternArray)) + 1};
end

mu = zeros(1, responseNum);
sigma = zeros(responseNum, responseNum);
sigma(:,:) = correlation;
for i = 1:responseNum
    sigma(i,i) = 1;
end
sigma = sigma .* noiseLevel;

dataset = cell(datasetSize, 2);
for i = 1:datasetSize
    e = mvnrnd(mu, sigma);
    X = random('Normal', 0, 1, predictorSize);
    for j = 1:responseNum
        Y = sum(sum(models{j} .* X)) + e(j);
        dataset{i, j} = Y;
    end
    dataset{i, responseNum+1} = X;
end

end

