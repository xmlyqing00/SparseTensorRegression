% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me

clc;
clear all;

y = tensor(rand(4,3,2));
components = DecomposeTensor(y, 10);
disp(y);
disp(components);
x = ComposeTensor(components);
disp(x);
