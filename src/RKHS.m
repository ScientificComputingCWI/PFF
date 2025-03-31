function kernel = RKHS(x,y,p)
% x, y vectors

type = p.type;

switch type
    case 'exp'
        A  = p.A;
        if (length(x)==length(y))
            % A should be length(x)*length(y)
            kernel = exp(-(1/2)*(x-y).'*A*(x-y));
        elseif (length(x)==1 && length(y)>1)
            % A should be 1x1, kernel is length(y)
            kernel = exp(-(1/2)*(x-y).*A.*(x-y));
        elseif (length(y)==1 && length(x)>1)
            % A should be 1x1, kernel is length(y)
            kernel = exp(-(1/2)*(x-y).*A.*(x-y));
        end
    case 'linear'
end

end