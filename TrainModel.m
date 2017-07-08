function [ models ] = TrainModel( lambda, rank, trainingSet, validationSet )
%TrainModel Train the models by trainingSet and validationSet.
%   Formula: Y = BX + e
%   Given Y and X, train model B.

startIter = 201;
iterTotal = 2000;
learningRate = 1e-3;
minLearningRate = 1e-7;
overfittingRate = 2;

[trainingSetSize, cols] = size(trainingSet);
responseNum = cols - 1;
dims = size(trainingSet{1, cols});
D_way = length(dims);

% Initialize the random models.
%models = InitModels(responseNum, D_way, dims, rank);

% Load models from files.
models = LoadModels(startIter, responseNum);

% Set the models as the target models.
% load('data/pattern3.mat', 'pattern');
% models = cell(1, 2);
% models{1} = DecomposeTensor(tensor(pattern), rank);
% models{2} = models{1};

trainingFuncValue = CalcObjFunc(models, lambda, trainingSet);
validationFuncValue = CalcObjFunc(models, lambda, validationSet);

for iter = startIter+1:iterTotal
    
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
    
    for dataIndex = 1:trainingSetSize
        
        trainingDataTensor = tensor(trainingSet{dataIndex, cols});
        
        for q = 1:responseNum
            diff0 = -2 * (trainingSet{dataIndex, q} - ttt(modelsTensor{q}, trainingDataTensor, 1:D_way));
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
                        
                        diff = diff0 * ttt(diffTensor, trainingDataTensor, 1:D_way);
                        modelsGrad{q}{d}(i,r) = modelsGrad{q}{d}(i,r) + diff;
                        
                    end
                end
            end
        end
    end
    
%     for q = 1:responseNum
%         for d = 1:D_way
%             modelsGrad{q}{d} = modelsGrad{q}{d} / trainingSetSize;
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
    
    disp(modelsGrad{1}{2}(2,1));
    
    newModels = models;
    for q = 1:responseNum
        for d = 1:D_way
            newModels{q}{d} = models{q}{d} - learningRate * modelsGrad{q}{d};
        end
    end
    
    preTrainingFuncValue = trainingFuncValue;
    trainingFuncValue = CalcObjFunc(newModels, lambda, trainingSet);  
    disp(learningRate);
    disp(preTrainingFuncValue);
    disp(trainingFuncValue);
    
    while trainingFuncValue >= preTrainingFuncValue
        
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
        disp(learningRate);
        disp(preTrainingFuncValue);
        disp(trainingFuncValue);
    
    end

    if trainingFuncValue < preTrainingFuncValue
        models = newModels;
    else
        break;
    end

    SaveTrainingStatus(iter, models);

    preValidationFuncValue = validationFuncValue;
    validationFuncValue = CalcObjFunc(models, lambda, validationSet);
    disp('validation');
    disp(preValidationFuncValue);
    disp(validationFuncValue);
    if validationFuncValue >= overfittingRate * preValidationFuncValue
        break;
    end
    
    disp(' ');
    
end

end

