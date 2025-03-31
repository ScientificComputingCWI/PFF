% RKHS example

% defining property:
% f(x) = <f,K(.,x)>
% where f(x) = sum_i \alpha_i K(x_i,x)

clearvars
clc
close all

%% example 1: choose coefficients alpha, get f(x)
x_train = [-2;0;2];
alpha   = [1;-0.5;0.8];
N       = length(alpha);
K_type  = 'exp';
sigma   = 1;
p.A     = 1/sigma^2;
p.type  = K_type;
n_eval  = 40;
K = RKHS(x_train,0,p);

x_eval = linspace(-4,4,n_eval)';
f_eval = zeros(n_eval,1);

% for eacht test point x_eval, compute sum_i alpha_i K(x_i,x_eval)
for i=1:n_eval
    f_eval(i) = linear_combination(x_eval(i),x_train,alpha,p);
end

figure
plot(x_eval,f_eval,'x-');
title('coefficients and kernel given, show f(x)')

%% inner product <alpha K(), beta K()>
% check the reproducing property:
x_test = 0.2;
M = 6;
e = zeros(M,1);
e(1) =1;
x_test_vec = [x_test;zeros(M-1,1)];

% evaluate inner product <f(x),K(x_test,x)> and check if it evaluates to
% f(x_test); for this we evaluate the inner product with g(x) =
% 1*K(x_test,x) + 0*...
% <f(x),K(x_test,x)>:
inner_product(alpha,x_train,e,x_test_vec,p)
% f(x_test):
f_test = linear_combination(x_test,x_train,alpha,p)

%% find coefficients alpha given some function f(x)
x_train = [0;1;2];
f_train = sin(x_train);
N_train = length(x_train);

% get coefficients from solving K \alpha = f
K_mat = zeros(N_train,N_train);
for i=1:N_train
    for j=1:N_train
        K_mat(i,j) = RKHS(x_train(i),x_train(j),p);
    end
end
alpha = K_mat\f_train;

x_eval = linspace(-4,4,n_eval)';
f_eval = zeros(n_eval,1);

% for eacht test point x_eval, compute sum_i alpha_i K(x_i,x_eval)
for i=1:n_eval
    f_eval(i) = linear_combination(x_eval(i),x_train,alpha,p);
end

figure
plot(x_eval,f_eval,'x-');
hold on
plot(x_train,f_train,'o');
title('training points and kernel given, find coefficients and show f(x)')


