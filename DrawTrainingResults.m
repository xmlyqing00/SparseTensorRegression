function [] = DrawTrainingResults()
%DrawTrainingResults Draw the training results and the estimated models.

load('training/trainingResults.mat', 'trainingResults');
[iterTotal, cols] = size(trainingResults);
fig = figure;

subplot(3, 2, [1, 2]);
hold on;
plot(trainingResults(:,1), trainingResults(:,2), 'r-');
plot(trainingResults(:,1), trainingResults(:,3), 'g-');
hold off;
legend('Training Dataset', 'Validation Dataset');
title('Objective Function Value');

subplot(3, 2, 3);
pattern = imread('data/pattern1.bmp');
imshow(pattern);
title('Generated Pattern');

subplot(3, 2, 4);
model = imread(['training/model_', num2str(iterTotal), '_2.bmp']);
imshow(model);
title('Estimated Pattern');

subplot(3, 2, 5);
pattern = imread('data/pattern2.bmp');
imshow(pattern);
title('Generated Pattern');

subplot(3, 2, 6);
model = imread(['training/model_', num2str(iterTotal), '_1.bmp']);
imshow(model);
title('Estimated Pattern');

imwrite(fig, 'training/summary.bmp');

end

