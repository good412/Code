function [n, M, nClass, labels, rlabels] = load_gn(dataset)

cd =['data' filesep ];

n=128;
str1=[cd, dataset  ,'.dat']; %network filename

A=load(str1);
m=size(A,1);
M=sparse(A(:,1),A(:,2),ones(m,1),n,n);
M=full(M);
M=M+M';
[x,y,z]=find(M);
M=[x,y];

labels = [ones(1,32),ones(1,32)*2,ones(1,32)*3,ones(1,32)*4];

nClass = max(labels);
rlabels = cell(1,nClass);
[Y,I] = sort(labels);
for i=1:nClass
    rlabels{1,i} = I(Y==i);
end

end