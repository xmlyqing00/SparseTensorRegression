% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me

function [ composedTensor ] = ComposeTensor( components )
%ComposeTensor Compose a tensor by components.

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

