function [ models ] = TrainModelDerivation( lambda, rank, trainingSet, ...
    validationSet )
%TrainModelDerivation Train the model by the analytical solution and
%derivation iteration.

iterStart = 0;
iterTotal = 100;
overfittingRate = 1.1;
shakyRate = 1.1;

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

trainingFuncValue = CalcObjFunc(models, lambda, trainingSet);
minValidationFuncValue = CalcObjFunc(models, lambda, validationSet);

XMatrices = cell(trainingSetSize, D_way);
for dataIndex = 1: trainingSetSize
    for d = 1:D_way
        cdims = 1:D_way;
        cdims(d) = [];
        XMatrices{dataIndex, d} = tenmat(trainingSet{dataIndex, cols}, cdims, 't');
    end
end

XMatricesTilde = cell(trainingSetSize, responseNum, D_way);
for iter = iterStart+1:iterTotal
    
    disp(iter);
    
    newModels = models;
    
    for q = 1:responseNum
        for d = 1:D_way
            khatriraoResult = ones(1, rank);
            for d2 = D_way:-1:1
                if d2 ~= d
                    khatriraoResult = khatrirao(khatriraoResult, models{q}{d2});
                end
            end
            for dataIndex = 1:trainingSetSize
                XMatricesTilde{dataIndex, q, d} = XMatrices{dataIndex, d} * khatriraoResult;
            end
        end
    end
    
    
            for dataIndex = 1:trainingSetSize
        
            
            diff0 = -2 * ...
                (trainingSet{trainingSetIndex, q} - ...
                ttt(modelsTensor{q}, trainingDataTensor, 1:D_way));
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
                        
                        diff = diff0 * ttt(diffTensor, ...
                            trainingDataTensor, 1:D_way);
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
                    modelsGrad{q}{d}(i,r) = modelsGrad{q}{d}(i,r) + ...
                        lambda * diff;
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
    
    preTrainingFuncValue = trainingFuncValue;
    trainingFuncValue = CalcObjFunc(newModels, lambda, trainingSet);  
    
    while trainingFuncValue >= preTrainingFuncValue
        
        learningRate = learningRate / 2;
        if learningRate < minLearningRate
            break;
        end
        for q = 1:responseNum
            for d = 1:D_way
                newModels{q}{d} = models{q}{d} - ...
                    learningRate * modelsGrad{q}{d};
            end
        end

        trainingFuncValue = CalcObjFunc(newModels, lambda, trainingSet); 
        
    
    end

    if trainingFuncValue < shakyRate * preTrainingFuncValue
        models = newModels;
    else
        break;
    end
    
    disp('training');
    disp(learningRate);
    disp(preTrainingFuncValue);
    disp(trainingFuncValue);

    validationFuncValue = CalcObjFunc(models, lambda, validationSet);
    disp('validation');
    disp(minValidationFuncValue);
    disp(validationFuncValue);
    
    SaveTrainingStatus(iter, models, trainingFuncValue, validationFuncValue);
    
    if validationFuncValue >= overfittingRate * minValidationFuncValue
        break;
    else
        minValidationFuncValue = min(minValidationFuncValue, ...
            validationFuncValue);
    end
    
    disp(' ');
    
end


end

