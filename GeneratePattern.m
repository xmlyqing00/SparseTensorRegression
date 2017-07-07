function [ returnStatus ] = GeneratePattern( patternSize )
%GeneratePattern Generate default patterns and save them to .mat files.
%   If the pattern files exist, then skip this function.

%Check pattern files.
if exist('data/pattern1.mat', 'file') ~= 0 && exist('data/pattern2.mat', 'file') ~= 0 && exist('data/pattern3.mat', 'file') ~= 0
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

%Pattern small
pattern = zeros(3, 3);
pattern(3,2) = 1;
pattern(2,3) = 1;
imwrite(pattern, 'data/pattern3.bmp');
save('data/pattern3.mat', 'pattern');

returnStatus = true;

end

