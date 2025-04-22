function residual = PFF(~,x,param)

% state x: matrix of Np particles (rows) each of dimension N, so Np*N matrix
% implementation so far tested for N=1

Np = size(x,1); % number of particles

d = param.d;
q = param.q;
p = param.p;

% d: observation
% kernel matrix A
% observation operator H, in this case simply y(x) and derivative J = dH/dx
% = dy/dx
% likelihood weight R
% parameters p

% see also Algorithm 1 in Hu & van Leeuwen


% initialize residual
res = zeros(Np,1);


A  = p.PFF.A;
fac = p.PFF.fac;

% R  = p.ll.R;
B  = p.ll.B;
xb = p.ll.xb;
% xb = mean(x,1); % background state for prior, take mean over particles
grad_log_p = zeros(Np,1);

% precompute the log posterior term, so that the model is only evaluated Np
% times instead of Np^2 times
for i=1:Np

    % evaluate model and get its Jacobian
    [y,J] = model(x(i),q,p);
    
    % evaluate gradient of log likelihood
    [~,dlogp_dx] = likelihood(y,d,p);

    % note that if J is a scalar then J.'=J
    % evaluate entire gradient of log posterior =  grad. log likelihood  +
    % grad log prior
    grad_log_p(i) = J*dlogp_dx - B\(x(i,:) - xb); 

end

% vectorized form
for i=1:Np

    kernel  = RKHS(x(:,:),x(i,:),p.PFF);
    % testing a factor of 2 by incorporating d/dtau of the posterior
    res(i)  = sum(kernel .* ( -(A.'*(x(:,:) - x(i,:))) + fac*grad_log_p ));

end

% non-vectorized form
% for i=1:Np
% 
% 
%     for j=1:Np
% 
%         res(i) = res(i) + RKHS(x(j,:),x(i,:),p.PFF) * ...
%                  ( -(A.'*(x(j,:) - x(i,:))) + ... % div K
%                   grad_log_p(j) );  % grad log post
% 
% 
%     end
% end

% divide by number of particles and scale
% residual = B*res/Np;
residual = res/Np;
    