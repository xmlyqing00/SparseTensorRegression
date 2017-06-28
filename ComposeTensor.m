function [ composedTensor ] = ComposeTensor( components )
%ComposeTensor Compose vectors to a tensor.
%   Input argument must be a cp_als result variable.

rank = length(components.lambda);
D_way = length(components.U);
dims = [];
for i = 1:D_way
    [rows, cols] = size(components.U{i});
    dims(i) = rows;
end

composedArray = zeros(dims);
for r = 1:rank
    tmpArray = 1;
    for i = 1:D_way
        tmpArray = kron(components.U{i}(:,r), tmpArray);
    end
    tmpArray = reshape(tmpArray, dims);
    tmpArray = components.lambda(r) .* tmpArray; 
    composedArray = composedArray + tmpArray;
end

composedTensor = tensor(composedArray);

end

