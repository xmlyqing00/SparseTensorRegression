function [ composedTensor ] = ComposeTensor( components )
%ComposeTensor Compose a tensor by components.
%   Components is a cell of matrixs. The matrix size is [dim, rank].
%   The composedTensor is the outer production of the input components.

D_way = length(components);
dims = [];
for i = 1:D_way
    [rows, rank] = size(components{i});
    dims(i) = rows;
end

composedArray = zeros(dims);
for r = 1:rank
    tmpArray = 1;
    for i = 1:D_way
        tmpArray = kron(components{i}(:,r), tmpArray);
    end
    tmpArray = reshape(tmpArray, dims);
    composedArray = composedArray + tmpArray;
end

composedTensor = tensor(composedArray);

end

