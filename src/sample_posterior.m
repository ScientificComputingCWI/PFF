%% compute posterior

% see Evensen (2022), chapter 18
% https://github.com/geirev/EnKF_scalar

clearvars
close all

test_case = 3;

%% load settings
settings = load_test_case(test_case);

% C's are covariance matrix entries, note covariance matrix ~ st.dev.^2
mu_x = settings.mu_x; % prior on x
C_xx = settings.C_xx;
mu_q = settings.mu_q; % prior on model error
C_qq = settings.C_qq;
mu_d = settings.mu_d; % observation mu and covariance
C_d  = settings.C_d;

% model settings
beta       = settings.beta;
model_type = settings.model_type;

%% samples for evaluating Bayes directly
N_samples = 1e6;

%% plotting settings
N_plot    = 1000;
save_fig = 0;

%% PFF settings
% number of particles
N_PFF      = 200;
% kernel
PFF_type   = 'exp';
PFF_kernel = pi; % this corresponds to Evensen (2022), chapter 18, where 2/pi is used for C. in our case, we set A=2/C
% time integration span for PFF
T_start    = 0;
T_end      = 100;
integration_type = 'FE'; % 'FE' or 'ode23'
Nt         = 1000; % number of steps for PFF in case of FE
tol_PFF    = 1e-3; % stopping criterion for time integration in case of Matlab's ode solver
fac        = 1; % additional factor to scale grad_log_p term

%% generate samples

% uniform samples (at which the prior will be evaluated)
x_prior_left  = mu_x-3*sqrt(C_xx);
x_prior_right = mu_x+3*sqrt(C_xx);
x_prior_uni  = linspace(x_prior_left,x_prior_right,N_samples)';
% q_prior_uni  = linspace(mu_q-3*sqrt(C_qq),mu_q+3*sqrt(C_qq),N_samples)';
% q_prior_uni  = q_prior_uni(randperm(N_samples));
% q_prior_uni  = mu_q + sqrt(C_qq)*randn(N_samples,1);

% samples of the actual prior on x
x_prior = mu_x + sqrt(C_xx)*randn(N_samples,1);

% samples of the model error on q
% option 1: take N_samples for the model error; this seems to be what's
% happening in https://github.com/geirev/EnKF_scalar/blob/master/src/m_enstein.F90
% q_prior = mu_q + sqrt(C_qq)*randn(N_samples,1);
% option 2: take a single sample model error:
q_prior = mu_q + sqrt(C_qq)*randn;

% observation samples
% obs     = mu_d + sqrt(C_d)*randn(N_data,1);
% generate a synthetic observation by perturbing the model
% obs    = mu_d + sqrt(C_d)*randn(1,1); %model(mu_d,mu_q,p) + sqrt(C_d);
obs    = mu_d;
N_data = 1;


%% set parameters
% set likelihood
p.ll.R     = C_d; % covariance on observation
p.ll.B     = C_xx; % covariance on prior

% set model parameters
p.model.beta = beta;
p.model.model_type = model_type;


%% evaluate posterior by calculating likelihood using samples from prior

% allocate posterior
post  = zeros(N_samples,1);

for i=1:N_samples

    % evaluate model on this sample
    y  = model(x_prior(i),q_prior,p);

    % initialize likelihood
    ll = 1;   

    for j=1:N_data % for large N_data the multiplication will lead to machine zero
        % obs(j) = y + sqrt(C_d);
        % p(d | x, q)
        ll = ll*likelihood(y,obs(j),p);
    end

    post(i) = ll; % don't need to multiply by prior, as this is effectively incorporated in the prior sampling

end

% normalize
post_norm = post; % ./ sum(post);
% Resample from posterior using weighted sampling (not needed per se)
% posterior_indices = randsample(1:N_samples, N_samples, true, post_norm);
% posterior_samples = prior(posterior_indices);

%% evaluate posterior by calculating likelihood using uniform samples

prior_uni = (1/sqrt(2*pi*C_xx))*exp(-(1/2)*(x_prior_uni - mu_x).^2/C_xx);

% allocate posterior
post_uni  = zeros(N_samples,1);

for i=1:N_samples

    % evaluate model on this sample
    y  = model(x_prior_uni(i),q_prior,p);

    % initialize likelihood
    ll = 1;   

    for j=1:N_data % for large N_data the multiplication will lead to machine zero
        % obs(j) = y + sqrt(C_d);
        % p(d | x, q)
        ll = ll*likelihood(y,obs(j),p);
    end

    post_uni(i) = ll * prior_uni(i); 

end

% normalize
evidence      = sum(post_uni)*(x_prior_right - x_prior_left)/N_samples;
post_uni_norm = post_uni/evidence;

% Resample from posterior using weighted sampling (not needed per se)
% posterior_indices = randsample(1:N_samples, N_samples, true, post_norm);
% posterior_samples = prior(posterior_indices);


%% evaluate posterior by using particle flow filter

% p.PFF.A    = 0.1; % choice for kernel parameter -> to be tuned
% note that A here is C^-1 in the book of van Leeuwen
p.PFF.A    = PFF_kernel; % 10./p.ll.B; % see Hu & van Leeuwen, eqn. (18), and eqn (18.4) in Evensen (2022)
p.PFF.type = PFF_type;
p.PFF.fac  = fac;
% prior only on x
x_prior_PFF = mu_x + sqrt(C_xx)*randn(N_PFF,1);
% model error on q
q_prior_PFF = q_prior; %mu_q + sqrt(C_qq)*randn(N_PFF,1);

p.ll.xb     = mu_x; %mean(x_prior_PFF,1);

% set up structure to evaluate PFF
param.d = obs;
param.q = q_prior_PFF;
param.p = p; 
param.tol = tol_PFF; % stopping tolerance for time integration
res_initial = PFF(T_start,x_prior_PFF,param);

% residual of prior
% PFF(x_prior_PFF,obs,q_prior_PFF,p);

% check residual of posterior from Bayes
% res_Bayes = PFF(T_end,post_norm,param);

% solve a non-linear system (this will usually not work)
% x_PFF = fsolve(@(x) PFF(T_end,x,param),x_prior_lin);

% time integration

switch integration_type

    case 'ode23'
        t_span = [0 T_end];
        options = odeset('RelTol',1e-3,'AbsTol',1e-3,'Events', @(t,x)steadyStateEvent(t,x,param)); % relax the tolerances as only interested in steady state
        % solve ODE with standard Matlab library
        % as initial condition use the prior
        % output is of dimension size(t_out)*Np
        [t_out,x_out] = ode23(@(t,x) PFF(t,x,param), t_span, x_prior_PFF, options);
        if (t_out(end)<T_end)
            disp(['residual tolerance satisfied at t=' num2str(t_out(end))]);
        end
        DKL   = zeros(Nt+1,1);
        for k=2:length(t_out)
            DKL(k) = kl_divergence(x_out(k-1,:),x_out(k,:));
        end

    case 'FE' % Forward Euler
        dt    = T_end/Nt;
        x_out = zeros(Nt+1,N_PFF);
        x     = x_prior_PFF;
        x_out(1,:) = x;
        DKL_inst   = zeros(Nt+1,1);

        t  = 0;
        for k=2:Nt+1
            rhs = PFF(t,x,param);
            % rhs = PFF(x,obs,q_prior_PFF,p);
            x_new = x + dt*rhs; % forward Euler update
            t     = t + dt;
            DKL_inst(k)   = kl_divergence(x,x_new);
            x     = x_new;
            x_out(k,:) = x_new;
        end
        t_out = 0:dt:T_end;


end
x_final = x_out(end,:);

DKL_post   = zeros(Nt+1,1);
for k=1:Nt
    DKL_post(k) = kl_divergence(x_out(k,:),x_final);
end

% get KDE of the particles
[pdf_vals, x_vals] = ksdensity(x_final);


%% plot particles positions and residuals

figure
n_skip = ceil(N_PFF/100); % plot at most 100 particles
plot(t_out,x_out(:,1:n_skip:end))
xlabel('pseudo time')
ylabel('particle positions')
grid on
box on
set(gca,'FontSize',14)

res_final = PFF(T_end,x_final.',param);

figure
semilogy(abs(res_initial));
hold on
semilogy(abs(res_final));
legend('initial residual','final residual')
xlabel('particle number')
ylabel('residual')
grid on
box on
set(gca,'FontSize',14)


%% plot densities

figure
hold on
plot(x_prior_uni,prior_uni, 'LineWidth', 2);
% plot(prior,post_norm,'x');
% [~, edges] = histcounts(prior, 50); % Get histogram bin edges for alignment
% bin_centers = (edges(1:end-1) + edges(2:end)) / 2; % Calculate bin centers
% bar(bin_centers, post_norm, 'histc');
[posterior_density, posterior_bins] = ksdensity(x_prior, 'Weights', post_norm,'NumPoints',500);
plot(posterior_bins, posterior_density, 'LineWidth', 2);

% we could do a kde 
% [posterior_uni_density, posterior_uni_bins] = ksdensity(x_prior_uni, 'Weights', post_uni,'NumPoints',500);
% plot(posterior_uni_bins, posterior_uni_density, 'LineWidth', 2);
plot(x_prior_uni,post_uni_norm,'LineWidth',2);


plot(x_vals, pdf_vals, 'LineWidth', 2); % KDE plot of PFF


% histogram(posterior_samples, 50, 'Normalization', 'pdf');

% xlim([-2 3]);
xlim([mu_x-3*sqrt(C_xx),mu_x+3*sqrt(C_xx)]);
legend('prior','posterior by sampling prior and kde','posterior by sampling uniform', 'posterior PFF','Location','northwest'); %,'posterior resampled')
grid on
box on
set(gca,'FontSize',14)

if (save_fig)
    addpath('/Users/sanderse/Library/CloudStorage/Dropbox/work/Programming/libs/export_fig-master/')
    export_fig(['posterior_testcase_' num2str(test_case)],'-pdf','-transparent');
end


%% plot KL
figure
hold on
plot(t_out(2:end),DKL_inst(2:end),'LineWidth',2);
plot(t_out(2:end),DKL_post(1:end-1),'LineWidth',2);

grid on
box on
set(gca,'FontSize',14)
xlabel('pseudo time')
ylabel('KL divergence')
legend('DKL(x^n,x^{n+1}','DKL(x^n,x^N)')


