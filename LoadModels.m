function [ models ] = LoadModels( iterStart, responseNum )
%LoadModels Load models from files to continue training process.

models = cell(1, responseNum);
for q = 1:responseNum
    loadStruct = load(['training/model_', num2str(iterStart), '_', num2str(q), '.mat']);
    models{q} = loadStruct.model;
end

end

