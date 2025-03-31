function [value, isterminal, direction] = steadyStateEvent(t, x, param)
    dx = PFF(t,x,param); 
    tol = param.tol;
    value = max(abs(dx)) - tol;   % tol = e.g., 1e-6
    isterminal = 1;           % stop integration when value = 0
    direction = 0;
end