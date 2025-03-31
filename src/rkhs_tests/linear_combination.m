function y = linear_combination(x_eval,x_train,alpha,p)

% length alpha should correspond to length x_train
N = length(alpha);
% y = zeros(N,1);
y = 0;
for i=1:N
    y = y + alpha(i).*RKHS(x_train(i),x_eval,p);
end


end