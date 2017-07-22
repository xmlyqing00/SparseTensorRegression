function [ returnStatus ] = GeneratePattern( patternSize, override )
%GeneratePattern Generate default patterns and save them to .mat files.
%   Parameters:
%       patternSize: The size of predictor.
%       override: The boolean flag decides whether to re-create new datasets.
%       
%   If the override falg is false and the pattern files exist, then skip 
%   this function.
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

%Check pattern files.
if override == false && ...
    exist('data/pattern1.mat', 'file') ~= 0 && ...
    exist('data/pattern2.mat', 'file') ~= 0 && ...
    exist('data/pattern3.mat', 'file') ~= 0 && ...
    exist('data/pattern4.mat', 'file') ~= 0 && ...
    exist('data/pattern5.mat', 'file') ~= 0

    disp('Generate the patterns. Skip.');
    return;
end

%Pattern Cross
pattern = ones(patternSize);
pattern(30:35, 17:48) = 0;
pattern(17:48, 30:35) = 0;
imwrite(pattern, 'data/pattern1.bmp');
save('data/pattern1.mat', 'pattern');

%Pattern Triangle
pattern = ones(patternSize);
marginX = round(patternSize / 2);
marginY = 21;
for y = marginY:patternSize-marginY
    for x = marginX:patternSize-marginX
        pattern(y, x) = 0;
    end
    marginX = marginX - 1;
    if marginX < 9
        break;
    end
end
imwrite(pattern, 'data/pattern2.bmp');
save('data/pattern2.mat', 'pattern');

%Pattern five squares
pattern = ones(patternSize);
pattern(5:16, 11:22) = 0;
pattern(51:62, 5:16) = 0;
pattern(33:44, 21:32) = 0;
pattern(29:40, 49:60) = 0;
pattern(13:24, 44:55) = 0;
imwrite(pattern, 'data/pattern3.bmp');
save('data/pattern3.mat', 'pattern');

%Pattern hollow square
pattern = ones(patternSize);
pattern(17:20, 17:48) = 0;
pattern(45:48, 17:48) = 0;
pattern(17:48, 17:20) = 0;
pattern(17:48, 45:48) = 0;
imwrite(pattern, 'data/pattern4.bmp');
save('data/pattern4.mat', 'pattern');

%Pattern small
pattern = zeros(3, 3);
pattern(3,2) = 1;
pattern(2,3) = 1;
imwrite(pattern, 'data/pattern5.bmp');
save('data/pattern5.mat', 'pattern');

disp('Generate the patterns. Finish.');
returnStatus = true;

end

