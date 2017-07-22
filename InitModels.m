function [ models ] = InitModels( responseNum, D_way, dims, rank )
%InitModels Initialize the models by arguments.
%   Parameters:
%       responseNum: The number of responses.
%       D_way: The number of components.
%       dims: A array of dims of each component.
%       rank: The rank of tensor decomposition and composition.
%
%   Initialize the components of model by random values.
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

models = cell(1, responseNum);

for q = 1:responseNum
    models{q} = cell(1, D_way);
    for d = 1:D_way
        models{q}{d} = rand(dims(d), rank);
    end
end

end

