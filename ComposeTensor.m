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

for i = 1:D_way
    
end

end

