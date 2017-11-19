function create_rank_list( feature, fileName )
%CREATE_RANK_LIST 此处显示有关此函数的摘要
%   此处显示详细说明

[q_num, ~] = size(feature);
new_feature = convert3(feature);
% new_feature = vl_homkermap(feature', 1, 'kchi2', 'gamma', .5)' ;
for i=1:q_num
    d = repmat(new_feature(i,:),q_num,1);
    b = new_feature - d;
    b = abs(b);
    c = sum(b,2);
    [~, I] = sort(c);

    if I(1) ~= i
        for j = 2:q_num
            if I(j) == i
                I(j) = I(1);
                I(1) = i;
            end
        end
    end
    dlmwrite(fileName, [num2str(i) '.dat ' num2str(I(1:187)' - 1)], '-append', 'delimiter','');
end

end
function S=convert3(X)
b=sum(X,2);
c=size(X,2);
D=repmat(b, [1,c]);
S=X./D;
end
