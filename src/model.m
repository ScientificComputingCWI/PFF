function [y,dydx] = model(x,q,p)
% return model y and derivative dydx
p = p.model;

switch p.model_type

    case 0
        y    = p.beta*x + q;
        dydx = p.beta;
    case 1
        y    = x + p.beta*x.^3 + q;
        dydx = 1 + 3*p.beta*x.^2;
    case 2
        y    = 1 + sin(pi*x) + q;
        dydx = pi*cos(pi*x);
    otherwise
        error('wrong model specification')
end