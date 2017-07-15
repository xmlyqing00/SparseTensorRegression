function [ models ] = TrainModel( lambda, rank, trainingSet, validationSet )
%TrainModel Train the models by trainingSet and validationSet.
%   Formula: Y = BX + e
%   Given Y and X, train model B.

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

% Set the models as the target models.
% models = cell(1, 2);
% load('data/pattern2.mat', 'pattern');
% models{1} = DecomposeTensor(tensor(pattern), rank);
% load('data/pattern1.mat', 'pattern');
% models{2} = DecomposeTensor(tensor(pattern), rank);

minTrainingFuncValue = CalcObjFunc(models, lambda, trainingSet);
minValidationFuncValue = CalcObjFunc(models, lambda, validationSet);

for iter = iterStart+1:iterTotal
    
    disp(iter);
%     models{1}{2}(2,1) = models{1}{2}(2,1) + 1e-5;
%     t1 = CalcObjFunc(models, lambda, trainingSet);
%     models{1}{2}(2,1) = models{1}{2}(2,1) - 2e-5;
%     t2 = CalcObjFunc(models, lambda, trainingSet);
%     x = (t1 - t2) / (2e-5);
%     disp(x);
    
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
    
    for q = 1:responseNum
        for d = 1:D_way
            modelsGrad{q}{d} = modelsGrad{q}{d} / sampleSetSize;
        end
    end
    
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

