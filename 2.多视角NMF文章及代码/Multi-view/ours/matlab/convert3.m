function [ S ] = convert3( X )
%CONVERT3 此处显示有关此函数的摘要
%   此处显示详细说明
b=(sum(abs(X),2));
c=size(X,2);
S=[];
for i = 1:c
    S(:,i) = X(:,i)./b;
end
end

