function [ models ] = LoadModels( iterStart, responseNum )
%LoadModels Load models from files to continue training process.
%   Parameters:
%       iterStart: The integer indicates where to start.
%       responseNum: The number of responses.
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

models = cell(1, responseNum);
for q = 1:responseNum
    loadStruct = load(['training/model_', num2str(iterStart), '_', num2str(q), '.mat']);
    models{q} = loadStruct.model;
end

end

