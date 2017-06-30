function [ components ] = DecomposeTensor( inputTensor, rank )
%DecomposeTensor Decompose the input tensor into components.

tensorComponents = cp_als(inputTensor, rank);
D_way = length(tensorComponents.U);
tensorComponents.lambda = tensorComponents.lambda .^ (1.0/D_way);
components = cell(1, D_way);
for i = 1:D_way
    for r = 1:rank
        tensorComponents.U{i}(:,r) = tensorComponents.U{i}(:,r) * tensorComponents.lambda(r);
    end
    components{i} = tensorComponents.U{i};
end

end

