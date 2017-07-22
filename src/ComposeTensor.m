function [ composedTensor ] = ComposeTensor( components )
%ComposeTensor Compose a tensor by the input components.
%   Parameters:
%       components: A cell of matrices. The number of matrices is D_way and 
%   each matrix size is [dims(d), rank].
%
%   The composedTensor is the outer production of the input components.
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

D_way = length(components);
dims = zeros(1, D_way);
for d = 1:D_way
    [dims(d), rank] = size(components{d});
end

composedArray = zeros(dims);
for r = 1:rank
    tmpArray = 1;
    for d = 1:D_way
        tmpArray = kron(components{d}(:,r), tmpArray);
    end
    tmpArray = reshape(tmpArray, dims);
    composedArray = composedArray + tmpArray;
end

composedTensor = tensor(composedArray);

end

