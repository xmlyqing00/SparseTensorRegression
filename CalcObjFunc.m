function [ funcValue ] = CalcObjFunc( models, lambda, dataset )
%CalcObjFunc Calculate the object function by given lambda and dataset
%   L = (Y - BX)^2 + \lambda*|B|

[datasetSize, cols] = size(dataset);
responseNum = cols - 1;
modelsTensor = cell(1, responseNum);
for i = 1:responseNum
    modelsTensor{i} = ComposeTensor(models{i});
end

D_way = length(models{1});

dims = zeros(D_way);
for i = 1:D_way
    [dims(i), rank] = size(models{1}{i});
end

funcValuePart1 = 0;
for dataIndex = 1:datasetSize
    for i = 1:responseNum
        funcValuePart1 = funcValuePart1 + ttt(modelsTensor{i}, tensor(dataset(i, cols)), 1:D_way);
    end
end

funcValuePart2 = 0;
for d = 1:D_way
    for r = 1:rank
        for i = 1:dims(d)
            tmp = 0;
            for j = 1:responseNum
                tmp = tmp + models{j}{d}(i,r) ^ 2;
            end
            funcValuePart2 = funcValuePart2 + tmp ^ 0.5;
        end
    end
end

funcValue = funcValue1 + lambda * funcValue2;

end

