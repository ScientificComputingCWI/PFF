function [p,dlogp_dx] = likelihood(y,d,param)
% y: "observation", usually y=y(x), 
% d: data

% note, Sigma is here the variance (scalar) or covariance matrix (matrix)
% p = exp(-(1/2)*(y - d).'*(Sigma\(y-d)));
Sigma = param.ll.R;

p = exp(-(1/2)*(d-y).' * (Sigma\(d-y)));

% also compute d log p/ dx but WITHOUT the adjoint dydx, this will be added in PFF.m
dlogp_dx = Sigma\(d-y);