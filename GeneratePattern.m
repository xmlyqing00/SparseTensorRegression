function [ returnStatus ] = GeneratePattern( patternSize )
%GeneratePattern Generate default patterns and save them to .mat files.
%   If the pattern files exist, then skip this function.

%Check pattern files.
if exist('pattern1.mat', 'file') ~= 0 && exist('pattern2.mat', 'file') ~= 0
    return;
end

%Pattern Cross
pattern = ones(patternSize);
pattern(30:35, 17:48) = 0;
pattern(17:48, 30:35) = 0;
imwrite(pattern, 'pattern1.bmp');
save('pattern1.mat', 'pattern');

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
imwrite(pattern, 'pattern2.bmp');
save('pattern2.mat', 'pattern');

returnStatus = true;

end

