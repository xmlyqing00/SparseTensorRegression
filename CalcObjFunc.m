function [ funcValue ] = CalcObjFunc( models, lambda, dataset )
%CalcObjFunc Calculate the objective function by the given models, lambda and dataset.
%   Parameters:
%       models: D_way components of tensor and its format is a cell of matrices. 
%       lambda: The coefficient of the penalty item.
%       dataset: A cell of two dimensions array. Each sample occupies a
%       line and the first q columns are the responses. The (q+1)th column
%       is the predictor.
%
%   funcValue = \sum(Y - model * dataset)^2 + \lambda * |model|
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

[datasetSize, cols] = size(dataset);
responseNum = cols - 1;
modelsTensor = cell(1, responseNum);
for q = 1:responseNum
    modelsTensor{q} = ComposeTensor(models{q});
end

D_way = length(models{1});

dims = zeros(1, D_way);
for d = 1:D_way
    [dims(d), rank] = size(models{1}{d});
end

funcValuePart1 = 0;
for dataIndex = 1:datasetSize
    for q = 1:responseNum
        loss = (dataset{dataIndex, q} - ttt(modelsTensor{q}, dataset{dataIndex, cols}, 1:D_way)) ^ 2;
        funcValuePart1 = funcValuePart1 + loss;
    end
end
% funcValuePart1 = funcValuePart1 / datasetSize;

funcValuePart2 = 0;
for d = 1:D_way
    for r = 1:rank
        for i = 1:dims(d)
            sumResult = 0;
            for q = 1:responseNum
                sumResult = sumResult + models{q}{d}(i,r) ^ 2;
            end
            funcValuePart2 = funcValuePart2 + sumResult ^ 0.5;
        end
    end
end

funcValue = funcValuePart1 + lambda * funcValuePart2;

end

