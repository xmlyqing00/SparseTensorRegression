function [ models ] = TrainModelDerivation( lambda, rank, trainingSet, validationSet )
%TrainModelDerivation Train the model by the analytical solution and derivation iteration.
%   Parameters:
%       lambda: The coefficent of the penalty term.
%       rank: A integer for CP decomposition / composition.
%       trainingSet: A set of samples for the model estimation.
%       validationSet: A set of sample for avoiding overfitting.
%
%   Formula: funcValue = \sum(Y - model * dataset)^2 + \lambda * |model|
%   Given Y, dataset and lambda.
%   We let the partial derivative to be zero and update the model B by its
%   closed-form solution.
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

iterStart = 0;
iterTotal = 10;
overfittingRate = 1.5;
shakyRate = 1.05;

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

XMat = cell(trainingSetSize, D_way);
for dataIndex = 1: trainingSetSize
    for d = 1:D_way
        cdims = 1:D_way;
        cdims(d) = [];
        XMat{dataIndex, d} = tenmat(trainingSet{dataIndex, cols}, cdims, 't');
    end
end

tic;
disp('Train the model. Running...');

for iter = iterStart+1:iterTotal

    newModels = models;
    for d = 1:D_way
        
        XMatTilde = cell(trainingSetSize, responseNum);
        for q = 1:responseNum
            khatriraoResult = ones(1, rank);
            for d2 = 1:D_way
                if d2 ~= d
                    khatriraoResult = khatrirao(khatriraoResult, newModels{q}{d2});
                end
            end
            rows = dims(d) * rank;
            for dataIndex = 1:trainingSetSize
                XMatTilde{dataIndex, q} = XMat{dataIndex, d} * khatriraoResult;
                XMatTilde{dataIndex, q} = reshape(XMatTilde{dataIndex, q}.data, [rows, 1]);
            end
        end
        
        mu = zeros(dims(d), rank);
        for i = 1:dims(d)
            for r = 1:rank
                mu(i, r) = 0;
                for q = 1:responseNum
                    mu(i, r) = mu(i, r) + newModels{q}{d}(i, r) ^ 2;
                end
                mu(i, r) = 1 / mu(i, r) ^ 0.5;
            end
        end
        
        rows = dims(d) * rank;
        diagMat = diag(reshape(mu, [rows, 1]));
        
        for q = 1:responseNum
            item1 = 0;
            item2 = 0;
            for dataIndex = 1:trainingSetSize
                item1 = item1 + XMatTilde{dataIndex, q} * XMatTilde{dataIndex, q}';
                item2 = item2 + trainingSet{dataIndex, q} * XMatTilde{dataIndex, q};
            end
            item1 = item1 - lambda / 2 * diagMat;
            newB = item1 \ item2;
            
            newModels{q}{d} = reshape(newB, [dims(d), rank]);
            
        end
        
    end
    
    trainingFuncValue = CalcObjFunc(newModels, lambda, trainingSet); 
    
    if trainingFuncValue < shakyRate * minTrainingFuncValue
        minTrainingFuncValue = min(minTrainingFuncValue, trainingFuncValue);
        models = newModels;
    else
        break;
    end
    
    validationFuncValue = CalcObjFunc(models, lambda, validationSet);
    
    disp(['    Iter: ', num2str(iter), '. TrainingSet: ', num2str(trainingFuncValue), '. ValidationSet: ', num2str(validationFuncValue)]);
    SaveTrainingStatus(iter, models, trainingFuncValue, validationFuncValue);
    
    if validationFuncValue >= overfittingRate * minValidationFuncValue
        break;
    end
    minValidationFuncValue = min(minValidationFuncValue, validationFuncValue);
    
end

disp('Train the model. Finish.');
toc;

end