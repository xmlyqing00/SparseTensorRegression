function [ models ] = TrainModelDerivation( lambda, rank, trainingSet, validationSet )
%TrainModelDerivation Train the model by the analytical solution and
%derivation iteration.

iterStart = 0;
iterTotal = 200;
overfittingRate = 1.1;
shakyRate = 1;

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

XMat = cell(trainingSetSize, D_way);
for dataIndex = 1: trainingSetSize
    for d = 1:D_way
        cdims = 1:D_way;
        cdims(d) = [];
        XMat{dataIndex, d} = tenmat(trainingSet{dataIndex, cols}, cdims, 't');
    end
end

for iter = iterStart+1:iterTotal
    
    disp(iter);

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

    disp('training');
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
        %break;
    end
    minValidationFuncValue = min(minValidationFuncValue, validationFuncValue);
    
    disp(' ');
    
end

end