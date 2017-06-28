% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me


clc;
clear all;

y = tensor(rand(4,3,2));
components = cp_als(y, 10);

disp(y);

x = ComposeTensor(components);
disp(x);
