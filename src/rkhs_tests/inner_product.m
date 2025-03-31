function d = inner_product(alpha,x_train,beta,y_train,p)

% inner product between two RKHS functions is defined through
% <f,g> = <sum_i alpha_i K(x_i,x),sum_j beta_j K(x_j,x)>
%       = sum_i sum_j alpha_i beta_j K(x_i,x_j)

M = length(alpha);
N = length(beta);

d = 0;

for i=1:M
    for j=1:N

        d = d + alpha(i)*beta(j)*RKHS(x_train(i),y_train(j),p);
    end
end