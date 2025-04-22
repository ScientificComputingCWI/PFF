function D_KL = kl_divergence(samples_p, samples_q, num_points)
% Compute KL divergence D_KL(P || Q) between two univariate sample sets
% using kernel density estimation and numerical integration.
%
% Inputs:
%   samples_p: samples from distribution P (1D array)
%   samples_q: samples from distribution Q (1D array)
%   num_points: number of points for evaluating PDFs (optional, default: 1000)
%
% Output:
%   D_KL: estimated KL divergence D_KL(P || Q)

if nargin < 3
    num_points = 1000;
end

% Determine evaluation range
min_val = min([samples_p(:); samples_q(:)]);
max_val = max([samples_p(:); samples_q(:)]);
x = linspace(min_val, max_val, num_points);

% Estimate densities using KDE
[p_pdf, ~] = ksdensity(samples_p, x, 'Function', 'pdf');
[q_pdf, ~] = ksdensity(samples_q, x, 'Function', 'pdf');

% Avoid division by zero and log of zero
epsilon = 1e-10;
p_pdf = max(p_pdf, epsilon);
q_pdf = max(q_pdf, epsilon);

% Compute KL divergence using trapezoidal numerical integration
D_KL = trapz(x, p_pdf .* log(p_pdf ./ q_pdf));

end
