function [ models ] = TrainModelGradDesc( lambda, rank, trainingSet, validationSet )
%TrainModelGradDesc Train the models with trainingSet by mini-batch gradient descending.
%   Parameters:
%       lambda: The coefficent of the penalty term.
%       rank: A integer for CP decomposition / composition.
%       trainingSet: A set of samples for the model estimation.
%       validationSet: A set of sample for avoiding overfitting.
%
%   Formula: funcValue = \sum(Y - model * dataset)^2 + \lambda * |model|
%   Given Y, dataset and lambda.
%   Train model B by mini-batch gradient descending.
%   B(t+1) = B(t) - learningRate * dfuncValue/dB
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

iterStart = 0;
iterTotal = 2000;
learningRate0 = 1e-3;
minLearningRate = 1e-6;
overfittingRate = 1.1;
shakyRate = 1.1;
sampleSetSize = 5;

[trainingSetSize, cols] = size(trainingSet);
responseNum = cols - 1;
dims = size(trainingSet{1, cols});
D_way = length(dims);

if iterStart == 0
    % Initialize the random models.
    models = InitModels(responseNum, D_way, dims, rank);
else
    % Load models from files.
    models = LoadModels(iterStart, responseNum);
end

minTrainingFuncValue = CalcObjFunc(models, lambda, trainingSet);
minValidationFuncValue = CalcObjFunc(models, lambda, validationSet);

for iter = iterStart+1:iterTotal
    
    disp(iter);
    
    modelsTensor = cell(1, responseNum);
    for q = 1:responseNum
        modelsTensor{q} = ComposeTensor(models{q});
    end
    
    modelsGrad = cell(1, responseNum);
    for q = 1:responseNum
        modelsGrad{q} = cell(1, D_way);
        for d = 1:D_way
            modelsGrad{q}{d} = zeros(dims(d), rank);
        end
    end
    
    sampleSetArray = round(rand(1, sampleSetSize) * (trainingSetSize - 1)) + 1;
    
    for sampleSetIndex = 1:sampleSetSize
        
        trainingSetIndex = sampleSetArray(sampleSetIndex);
        
        for q = 1:responseNum
            
            diff0 = -2 * (trainingSet{trainingSetIndex, q} - ttt(modelsTensor{q}, trainingSet{trainingSetIndex, cols}, 1:D_way));
            for r = 1:rank               
                for d = 1:D_way
                    for i = 1:dims(d)
                        
                        diffComponent = cell(1, D_way);
                        for d2 = 1:D_way
                            diffComponent{d2} = models{q}{d2}(:,r);
                        end
                        diffComponent{d}(:,1) = 0;
                        diffComponent{d}(i,1) = 1;
                        diffTensor = ComposeTensor(diffComponent);
                        
                        diff = diff0 * ttt(diffTensor, trainingSet{trainingSetIndex, cols}, 1:D_way);
                        modelsGrad{q}{d}(i,r) = modelsGrad{q}{d}(i,r) + diff;
                        
                    end
                end
            end
            
        end
    end
    
%     for q = 1:responseNum
%         for d = 1:D_way
%             modelsGrad{q}{d} = modelsGrad{q}{d} / sampleSetSize;
%         end
%     end
    
    for d = 1:D_way
        for r = 1:rank
            for i = 1:dims(d)
                
                sumResult = 0;
                for q = 1:responseNum
                    sumResult = sumResult + (models{q}{d}(i,r)) ^ 2;
                end
                sumResult = sumResult ^ 0.5;
                
                if sumResult == 0
                    continue;
                end
                
                for q = 1:responseNum
                    diff = models{q}{d}(i,r) / sumResult;
                    modelsGrad{q}{d}(i,r) = modelsGrad{q}{d}(i,r) + lambda * diff;
                end
                
            end
        end
    end
    
    disp('gradient');
    disp(modelsGrad{1}{2}(2,1));
    
    learningRate = learningRate0;
    newModels = models;
    for q = 1:responseNum
        for d = 1:D_way
            newModels{q}{d} = models{q}{d} - learningRate * modelsGrad{q}{d};
        end
    end
    
    trainingFuncValue = CalcObjFunc(newModels, lambda, trainingSet);  
    while trainingFuncValue >= minTrainingFuncValue
        
        learningRate = learningRate / 2;
        if learningRate < minLearningRate
            break;
        end
        for q = 1:responseNum
            for d = 1:D_way
                newModels{q}{d} = models{q}{d} - learningRate * modelsGrad{q}{d};
            end
        end

        trainingFuncValue = CalcObjFunc(newModels, lambda, trainingSet); 

    end

    disp('training');
    disp(learningRate);
    disp(minTrainingFuncValue);
    disp(trainingFuncValue);
    
    if trainingFuncValue < shakyRate * minTrainingFuncValue
        minTrainingFuncValue = min(minTrainingFuncValue, trainingFuncValue);
        models = newModels;
    else
        break;
    end
    
    validationFuncValue = CalcObjFunc(models, lambda, validationSet);
    disp('validation');
    disp(minValidationFuncValue);
    disp(validationFuncValue);
    
    SaveTrainingStatus(iter, models, trainingFuncValue, validationFuncValue);
    
    if validationFuncValue >= overfittingRate * minValidationFuncValue
        break;
    else
        minValidationFuncValue = min(minValidationFuncValue, validationFuncValue);
    end
    
    disp(' ');
    
end

end

