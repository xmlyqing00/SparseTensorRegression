function [ models ] = TrainModel( lambda, trainingSet, validationSet )
%TrainModel Train the models by trainingSet and validationSet.
%   Formula: Y = BX + e
%   Given Y and X, train model B.

iterTotal = 1000;
learningRate = 0.01;
minLearningRate = 0.000001;
[trainingSetSize, cols] = size(trainingSet);

responseNum = cols - 1;
dims = size(trainingSet{1, cols});
D_way = length(dims);
rank = 3;
models = InitModels(responseNum, D_way, dims, rank);

trainingFuncValue = CalcObjFunc(models, lambda, trainingSet);
validationFuncValue = CalcObjFunc(models, lambda, validationSet);

for iter = 1:iterTotal
    
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
    
    for dataIndex = 1:trainingSetSize
        for q = 1:responseNum
            diff0 = 2 * (trainingSet{dataIndex, q} - ttt(modelsTensor{q}, tensor(trainingSet{dataIndex, cols}), 1:D_way));
            for r = 1:rank               
                for d = 1:D_way
                    for i = 1:dims(d)
                        
                        singleComponent = cell(1, D_way);
                        for d2 = 1:D_way
                            singleComponent{d2} = models{q}{d2}(:,r);
                        end
                        singleComponent{d}(:,1) = 0;
                        singleComponent{d}(i,1) = 1;
                        singleTensor = ComposeTensor(singleComponent);
                        
                        diff = diff0 * -ttt(singleTensor, tensor(trainingSet{dataIndex, cols}), 1:D_way);
                        modelsGrad{q}{d}(i,r) = modelsGrad{q}{d}(i,r) + diff;
                        
                    end
                end
            end
        end
    end
    
    for q = 1:responseNum
        for d = 1:D_way
            modelsGrad{q}{d} = modelsGrad{q}{d} / trainingSetSize;
        end
    end
    
    for d = 1:D_way
        for r = 1:rank
            for i = 1:dims(d)
                sumResult = 0;
                for q = 1:responseNum
                    sumResult = sumResult + (models{q}{d}(i,r)) ^ 2;
                end
                for q = 1:responseNum
                    diff = models{q}{d}(i,r) / sumResult;
                    modelsGrad{q}{d}(i,r) = modelsGrad{q}{d}(i,r) + lambda * diff;
                end
            end
        end
    end
    
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
    
    if trainingFuncValue >= preTrainingFuncValue
        learningRate = learningRate / 2;
        if learningRate < minLearningRate
            break;
        end
    else
        models = newModels;
        
        t = ComposeTensor(models{1});
        tt = zeros(64, 64);
        tt(:) = t(:);
        figure;
        imshow(tt);
        
        preValidationFuncValue = validationFuncValue;
        validationFuncValue = CalcObjFunc(models, lambda, validationSet);
        if validationFuncValue >= preValidationFuncValue
            % break;
        end
    end
    
end

end

