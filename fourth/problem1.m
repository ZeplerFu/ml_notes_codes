% 自己改一改变量名和语句啥的
load('teapots.mat')
X = teapotImages;
m = mean(X,1);
X = X - ones(size(X,1),1)*m;
Cov = X'*X/size(X,1);
[E, D] = eig(Cov); 
d = diag(D);
[dum,ord] = sort(-d);
E = E(:,ord);
d = d(ord);
C = X*E;
E = E';
% 下面是绘图指令，自己写一下画十张的
n=3
for i=5:5:50
figure;
colormap gray;
imagesc(reshape(teapotImages(i,:),38,50));
figure;
colormap gray;
imagesc(reshape(m+C(i,1:n)*E(1:n,:),38,50));
end