function [ funcValue ] = CalcObjFunc( models, lambda, dataset )
%CalcObjFunc Calculate the object function by given lambda and dataset
%   L = (Y - BX)^2 + \lambda*|B|

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
        loss = (dataset{dataIndex, q} - ttt(modelsTensor{q}, tensor(dataset{dataIndex, cols}), 1:D_way)) ^ 2;
        funcValuePart1 = funcValuePart1 + loss;
    end
end

funcValuePart2 = 0;
for d = 1:D_way
    for r = 1:rank
        for i = 1:dims(d)
            tmp = 0;
            for q = 1:responseNum
                tmp = tmp + models{q}{d}(i,r) ^ 2;
            end
            funcValuePart2 = funcValuePart2 + tmp ^ 0.5;
        end
    end
end

funcValue = funcValuePart1 + lambda * funcValuePart2;

end

