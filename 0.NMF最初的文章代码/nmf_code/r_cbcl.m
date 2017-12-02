% r_cbcl

% load cbcldata
% sum(sum(V.*V))
for expnum=1:1
fname=['results3/smu3_sources49_rand1_experiment_' num2str(expnum),'.mat']
load(fname);
% W=W*diag(1./sqrt(sum(W.*W,1)));
H1=W';

% for j=1:25
%     b(j,:)=sort(H1(j,:),'descend');
% end
% figure(102)
% for jj=1:25
% plot(b(jj,:))
% hold on
% end
% b(1:2,1:50);
% kkkk


model_fig_image(H1,7,7,19,19)
% figure
% subplot(2,1,1)
% plot(F_obj(1,:))
% subplot(2,1,2)
% plot(F_obj(2,:))
% F_obj(1,180:200)
% r=25;
% for i=1:r
%     ws(1,i)=(sqrt(length(W(:,i)))-sum(abs(W(:,i)))/abs(sqrt(sum(W(:,i).*W(:,i)))))/(sqrt(length(W(:,i)))-1);
%     hs(1,i)=(sqrt(length(H(i,:)))-sum(abs(H(i,:)))/abs(sqrt(sum(H(i,:).*H(i,:)))))/(sqrt(length(W(:,i)))-1);
% end
% spa(expnum,:)=[mean(ws) mean(hs)];
end
% spa

