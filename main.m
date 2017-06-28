% Author:   Space Lyq
% Homepage: www.lyq.me
% Email:    root [at] lyq.me


clc;
clear all;

y = tensor(rand(4,3,2));
components = cp_als(y, 2);

disp(components.U{1}(1));

x = ComposeTensor(components);
disp(x);
