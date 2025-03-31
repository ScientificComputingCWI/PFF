function residual = PFF(t,x,param) %d,q,p)

% state x: matrix of Np particles (rows) each of dimension N, so Np*N matrix
% implementation so far tested for N=1

d = param.d;
q = param.q;
p = param.p;

% d: observation
% kernel matrix A
% observation operator H, so y = H(x) and derivative J = dHdx
% likelihood weight R
% parameters p

% see also Algorithm 1 in Hu & van Leeuwen

Np = size(x,1); %p_PFF.Np; % number of particles


% initialize residual
res = zeros(Np,1);

% this is the simplified form where a linear observation operator has been
% substituted
% for i=1:Np
% 
%     for j=1:Np
% 
%         res(i) = res(i) + RKHS(x(j,:),x(i,:),p) * ...
%                  ( -(A'*(x(j,:) - x(i,:))) + H'*R\(d-H*x(j,:)) );
% 
%     end
% 
% end


A  = p.PFF.A;
% R  = p.ll.R;
B  = p.ll.B;
xb = p.ll.xb;
% xb = mean(x,1); % background state for prior, take mean over particles
grad_log_p = zeros(Np,1);

% precompute the log posterior term, so that the model is only evaluated Np
% times instead of Np^2 times
for i=1:Np
    % J = p.model.beta;
    % J = (1+3*p.model.beta*x(i,:).^2);
     % J = pi*cos(pi*x(i,:));
    [y,J] = model(x(i),q(i),p);
    [~,dlogp_dx] = likelihood(y,d,p);
    % note that if J is a scalar then J.'=J
    grad_log_p(i) = J*dlogp_dx - B\(x(i,:) - xb); 

end

% vectorized form
for i=1:Np

    kernel  = RKHS(x(:,:),x(i,:),p.PFF);
    res(i)  = sum(kernel .* ( -(A.'*(x(:,:) - x(i,:)))+ grad_log_p ));

    % for j=1:Np
    % 
    %     % res(i) = res(i) + RKHS(x(j,:),x(i,:),p) * ...
    %     %          ( -(A'*(x(j,:) - x(i,:))) + H'*R\(d-H*x(j,:)) );
    % 
    %     res(i) = res(i) + RKHS(x(j,:),x(i,:),p.PFF) * ...
    %              ( -(A.'*(x(j,:) - x(i,:))) + ... % div K
    %               grad_log_p(j) );  % grad log post
    % 
    % 
    % end
end


% residual = B*res/Np;
residual = res/Np;
    