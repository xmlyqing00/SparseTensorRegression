function [ saveStatus ] = SaveTrainingStatus( iter, models, trainingFuncValue, validationFuncValue )
%SaveTrainingStatus Save the training status and variables to files during the training process.
%   Parameters:
%       iter: The current number of iterations.
%       models: The components of the estimated models.
%       trainingFuncValue: The objective function value of trainingSet.
%       validationFuncValue: The objective function value of validationSet.
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

responseNum = length(models);
D_way = length(models{1});
dims = zeros(1, D_way);
for d = 1:D_way
    [dims(d), ~] = size(models{1}{d});
end

for q = 1:responseNum
    t = ComposeTensor(models{q});
    tMat = zeros(dims);
    tMat(:) = t(:);
    imwrite(tMat, ['training/model_', num2str(iter), '_', num2str(q), '.bmp']);
    
    model = models{q};
    save(['training/model_', num2str(iter), '_', num2str(q), '.mat'], 'model');
end

newResult = [iter, trainingFuncValue, validationFuncValue];
if iter == 1
    trainingResults = newResult;
else
    load('training/trainingResults.mat', 'trainingResults');
    trainingResults = [trainingResults; newResult];
end
save('training/trainingResults.mat', 'trainingResults');

saveStatus = true;

end

