function [ drawStatus ] = DrawTrainingResults( responseNum )
%DrawTrainingResults Draw the training results and the estimated models.
%   Parameters:
%       responseNum: The number of responses.
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

load('training/trainingResults.mat', 'trainingResults');
[iterTotal, ~] = size(trainingResults);
figure;

subplot(responseNum + 1, 2, [1, 2]);
hold on;
plot(trainingResults(:,1), trainingResults(:,2), 'r-');
plot(trainingResults(:,1), trainingResults(:,3), 'g-');
hold off;
legend('Training Dataset', 'Validation Dataset');
xlabel('Iterations');
ylabel('Objective Function Value');

for q = 1:responseNum
    subplot(responseNum + 1, 2, 1 + q * 2);
    pattern = imread(['data/pattern', num2str(q), '.bmp']);
    imshow(pattern);
    title('Generated Pattern');

    subplot(responseNum + 1, 2, 2 + q * 2);
    model = imread(['training/model_', num2str(iterTotal), '_', num2str(q), '.bmp']);
    imshow(model);
    title('Estimated Pattern');
end

drawStatus = true;

end

