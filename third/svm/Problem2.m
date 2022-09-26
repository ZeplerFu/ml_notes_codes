load('dataset.mat')

n=size(X,1);
Y(Y==0)=-1;
n_train = 65;
randnum=randperm(size(X,1));
x_train = X(randnum(1:n_train),:);
y_train = Y(randnum(1:n_train),:);
size(x_train)
size(y_train)
x_test = X(randnum(n_train+1:end),:);
y_test = Y(randnum(n_train+1:end),:);
%  Values for ker: 'linear'  -
%                  'poly'    - p1 is degree of polynomial
%                  'rbf'     - p1 is width of rbfs (sigma)
%                  'sigmoid' - p1 is scale, p2 is offset
%                  'spline'  -
%                  'bspline' - p1 is degree of bspline
%                  'fourier' - p1 is degree
%                  'erfb'    - p1 is width of rbfs (sigma)
%                  'anova'   - p1 is max order of terms
global p
% ker = 'linear';
% ker = 'poly';
ker = 'rbf';
Cs = [0,1,2,3,4,inf];
kers = ['linear','poly','rbf'];
C=inf;
errors=zeros(6,3,9);
for i=1:6
    for j = 1:3
        for k = 1:9
            p=k;
            C=Cs(i);
            ker = kers(1,j);
            C;
            ker;
            [nsv, alpha, b0] = svc(x_train, y_train, ker, C);
            err = svcerror(x_train,y_train,x_test,y_test,ker,alpha,b0);
            errors(i,j,k)=err;
            err;
        end
    end
end
errors