function [p,dlogp_dx] = likelihood(y,d,param)
% y: "observation", usually y=y(x), 
% d: data

% note, Sigma is here the variance (scalar) or covariance matrix (matrix)
% p = exp(-(1/2)*(y - d).'*(Sigma\(y-d)));
Sigma = param.ll.R;


if (length(y)==length(d)) 
    % data point at each model evaluation, output is a scalar
    p = exp(-(1/2)*(d-y).' * (Sigma\(d-y)));
elseif (length(y)==1 && length(d)>1)
    % multiple data points for model evaluation, output is a vector
    p = exp(-(1/2)*(d-y).* (Sigma\(d-y)));
elseif (length(d)==1 && length(y)>1)
    % multiple evaluations per data point, output is a vector
    p = exp(-(1/2)*(d-y).* (Sigma\(d-y)));
end



% also compute d log p/ dx but WITHOUT the adjoint dydx, this will be added in PFF.m
dlogp_dx = Sigma\(d-y);