clear all;
YY=[];
Y=rand(10,1);
c = length(Y);
A=reshape(Y,1,c)
for i = reshape(Y,1,c)
    YY = [YY,Y==i];
end