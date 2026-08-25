function disp_moments_AL(M_, options_, oo_)
% DISP_MOMENTS_AL  Display AL moments and RE comparison. Mirrors disp_moments_SL.
%
%   AL moments : SIMULATED — MC mean over `replic` realizations, empirical CIs.
%   RE moments : THEORETICAL — unconditional second moments of the first-order
%                RE policy (native th_autocovariances).
% OUTPUT
%   1. Aggregate moments : AL mean/std + empirical CI  vs  RE std, ratio AL/RE
%   2. Persistence       : AL AR(1) + CI               vs  RE AR(1)
%   3. Belief dynamics    : perceived coefficients, time-mean and time-volatility
%                          (AL only; constant gain => perpetual fluctuation)
%
% Copyright © 2026 Dynare Team
%
% This file is part of Dynare.
%
% Dynare is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Dynare is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Dynare.  If not, see <https://www.gnu.org/licenses/>.

nvar = M_.endo_nbr;
A    = options_.learning.adaptive;

mom    = oo_.learning.adaptive.moments;
replic = mom.config.replic;

% RE THEORETICAL moments 
try
    [V_RE, AR1_RE] = re_moments(oo_.dr, M_, options_);
catch
    V_RE   = nan(nvar);
    AR1_RE = nan(nvar, 1);
end

% Table 1: Aggregate moments
fprintf('\nMOMENTS — Simulated AL (%d MC replications) vs Theoretical RE\n', replic);
fprintf(['  AL: Monte-Carlo mean, 2.5/97.5%% empirical CIs.   ' ...
    'RE: native theoretical moments.\n']);
fprintf('%-15s %8s %8s   [%7s ; %7s]   %8s %7s\n', ...
    'Variable', 'Mean', 'Std AL', '2.5%', '97.5%', 'Std RE', 'AL/RE');
fprintf('%s\n', repmat('-', 1, 78));
for iv = 1:nvar
    std_al = sqrt(oo_.var(iv, iv));
    std_re = sqrt(abs(V_RE(iv, iv)));
    if isfinite(std_re) && std_re > 1e-10
        fprintf('%-15s %8.4f %8.4f   [%7.4f ; %7.4f]   %8.4f %7.2f\n', ...
            M_.endo_names{iv}, oo_.mean(iv), std_al, ...
            mom.std_ci(iv,1), mom.std_ci(iv,2), std_re, std_al/std_re);
    else
        fprintf('%-15s %8.4f %8.4f   [%7.4f ; %7.4f]   %8s %7s\n', ...
            M_.endo_names{iv}, oo_.mean(iv), std_al, ...
            mom.std_ci(iv,1), mom.std_ci(iv,2), '-', '-');
    end
end

% Table 2: Persistence
fprintf('\nPERSISTENCE (First-order autocorrelation) — Simulated AL vs Theoretical RE\n');
fprintf('%-15s %8s   [%7s ; %7s]   %8s\n', ...
    'Variable', 'AR(1) AL', '2.5%', '97.5%', 'AR(1) RE');
fprintf('%s\n', repmat('-', 1, 58));
for iv = 1:nvar
    ar1_al = oo_.autocorr{1}(iv, iv);
    ar1_re = AR1_RE(iv);
    if isfinite(ar1_re)
        fprintf('%-15s %8.4f   [%7.4f ; %7.4f]   %8.4f\n', ...
            M_.endo_names{iv}, ar1_al, ...
            mom.autocorr_ci{1}(iv,1), mom.autocorr_ci{1}(iv,2), ar1_re);
    else
        fprintf('%-15s %8.4f   [%7.4f ; %7.4f]   %8s\n', ...
            M_.endo_names{iv}, ar1_al, ...
            mom.autocorr_ci{1}(iv,1), mom.autocorr_ci{1}(iv,2), '-');
    end
end

% Table 3: Belief dynamics 
fprintf('\nBELIEF DYNAMICS (AL only) — perceived coefficients over the sample\n');
if A.nb == 2
    fprintf('%-15s %12s %12s %12s %12s\n', 'Learned var', ...
        'const mean', 'const std', 'slope mean', 'slope std');
    fprintf('%s\n', repmat('-', 1, 66));
    for k = 1:A.nfwrd
        fprintf('%-15s %12.4f %12.4f %12.4f %12.4f\n', A.learn_names{k}, ...
            mom.beliefs.coef_mean(k,1), mom.beliefs.coef_std(k,1), ...
            mom.beliefs.coef_mean(k,2), mom.beliefs.coef_std(k,2));
    end
else
    fprintf('%-15s %12s %12s\n', 'Learned var', 'const mean', 'const std');
    fprintf('%s\n', repmat('-', 1, 42));
    for k = 1:A.nfwrd
        fprintf('%-15s %12.4f %12.4f\n', A.learn_names{k}, ...
            mom.beliefs.coef_mean(k,1), mom.beliefs.coef_std(k,1));
    end
end

% Projection facility diagnostic
fprintf('\nProjection facility: mean %.1f hits/run, max %d (out of %d periods).\n', ...
    mean(mom.pf_hits), max(mom.pf_hits), mom.config.periods);
fprintf('\n');

end


% Helper: fetch RE moments from Dynare native path
function [V, AR1] = re_moments(dr, M_, options_)
nvar = M_.endo_nbr;
ivar = (1:nvar)';
opts = options_;
if ~isfield(opts, 'ar') || opts.ar < 1
    opts.ar = 1;
end
[Gamma_y, ~] = th_autocovariances(dr, ivar, M_, opts, 1);  
V   = Gamma_y{1};
AR1 = diag(Gamma_y{2});
end