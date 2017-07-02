function [ models ] = TrainModel( lambda, trainingSet, validationSet )
%TrainModel Train the models by trainingSet and validationSet.
%   Formula: Y = BX + e
%   Given Y and X, train model B.

iterTotal = 1000;
learningRate = 0.1;
minLearningRate = 0.001;
[trainingSetSize, cols] = size(trainingSet);

responseNum = cols - 1;
dims = size(trainingSet(1, cols));
D_way = length(dims);
rank = 3;
models = InitModels(responseNum, D_way, dims, rank);

trainingFuncValue = CalcObjFunc(models, lambda, trainingSet);
validationFuncValue = CalcObjFunc(models, lambda, validationSet);

for iter = 1:iterTotal
    
    for i = 1:trainingSetSize
    end
    
    preTrainingFuncValue = trainingFuncValue;
    preValidationFuncValue = validationFuncValue;
    trainingFuncValue = CalcObjFunc(models, lambda, trainingSet);
    validationFuncValue = CalcObjFunc(models, lambda, validationSet);
    
    if traningFuncValue >= preTrainingFuncValue
        learningRate = learningRate / 2;
    end
    
    if learningRate < minLearningRate
        break;
    end
    
    if validataionFuncValue >= preValidationFuncValue
        break;
    end
    
end

end

