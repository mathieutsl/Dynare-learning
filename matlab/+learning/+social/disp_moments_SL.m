function disp_moments_SL(M_, options_, oo_)
% DISP_MOMENTS_SL  Display SL moments and RE comparaison 
%
%   SL moments : SIMULATED — MC mean over `replic` realizations, empirical CIs.
%   RE moments : THEORETICAL — unconditional second moments of the first-order
%                RE policy
% OUTPUT 
%   1. Aggregate moments : SL mean/std + empirical CI  vs  RE std, ratio SL/RE
%   2. Persistence       : SL AR(1) + CI               vs  RE AR(1)
%   3. Cross-section moments of beliefs (std, IQR, skewness) — SL only
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

nvar    = M_.endo_nbr;
id_fwrd = find(M_.lead_lag_incidence(3,:) > 0);

mom = oo_.learning.social.moments;
replic = options_.replic;

% RE theoretical L moments 
try
    [V_RE, AR1_RE] = re_moments(oo_.dr, M_, options_);
catch
    V_RE   = nan(nvar);
    AR1_RE = nan(nvar, 1);
end

% Table 1: Aggregate moments
fprintf('\nMOMENTS — Simulated SL (%d MC replications) vs Theoretical RE\n', replic);
fprintf(['  SL: Monte-Carlo mean, 2.5/97.5%% empirical CIs.   ' ...
    'RE: Native theoretical moments (th_autocovariances = stoch_simul periods=0).\n']);
fprintf('%-15s %8s %8s   [%7s ; %7s]   %8s %7s\n', ...
    'Variable', 'Mean', 'Std SL', '2.5%', '97.5%', 'Std RE', 'SL/RE');
fprintf('%s\n', repmat('-', 1, 78));

for iv = 1:nvar
    std_sl = sqrt(oo_.var(iv, iv));
    std_re = sqrt(abs(V_RE(iv, iv)));
    if isfinite(std_re) && std_re > 1e-10
        fprintf('%-15s %8.4f %8.4f   [%7.4f ; %7.4f]   %8.4f %7.2f\n', ...
            M_.endo_names{iv}, oo_.mean(iv), std_sl, ...
            mom.std_ci(iv, 1), mom.std_ci(iv, 2), ...
            std_re, std_sl / std_re);
    else
        fprintf('%-15s %8.4f %8.4f   [%7.4f ; %7.4f]   %8s %7s\n', ...
            M_.endo_names{iv}, oo_.mean(iv), std_sl, ...
            mom.std_ci(iv, 1), mom.std_ci(iv, 2), '-', '-');
    end
end

% Table 2 : Persistence
fprintf('\nPERSISTENCE (First-order autocorrelation) — Simulated SL vs Theoretical RE\n');
fprintf('%-15s %8s   [%7s ; %7s]   %8s\n', ...
    'Variable', 'AR(1) SL', '2.5%', '97.5%', 'AR(1) RE');
fprintf('%s\n', repmat('-', 1, 58));

for iv = 1:nvar
    ar1_sl = oo_.autocorr{1}(iv, iv);
    ar1_re = AR1_RE(iv);
    if isfinite(ar1_re)
        fprintf('%-15s %8.4f   [%7.4f ; %7.4f]   %8.4f\n', ...
            M_.endo_names{iv}, ar1_sl, ...
            mom.autocorr_ci{1}(iv, 1), mom.autocorr_ci{1}(iv, 2), ar1_re);
    else
        fprintf('%-15s %8.4f   [%7.4f ; %7.4f]   %8s\n', ...
            M_.endo_names{iv}, ar1_sl, ...
            mom.autocorr_ci{1}(iv, 1), mom.autocorr_ci{1}(iv, 2), '-');
    end
end

% Table 3 : Cross-section moments of beliefs
fprintf('\nCROSS-SECTIONAL BELIEF MOMENTS (SL only)\n');
fprintf('%-15s %10s %10s %10s\n', 'Learned var', 'Std', 'IQR', 'Skewness');
fprintf('%s\n', repmat('-', 1, 50));

if learning.is_active(options_, 'social')
    rows = learning.getopt(options_.learning.social, 'id_SL', [])';
else
    rows = find(mom.cross_section.std_mean > 1e-15)';
end
rows = rows(:)';

if isempty(rows)
    fprintf('%-15s\n', '(no learned variable)');
else
    for iv = rows
        fprintf('%-15s %10.6f %10.6f %10.4f\n', ...
            M_.endo_names{id_fwrd(iv)}, ...
            mom.cross_section.std_mean(iv), ...
            mom.cross_section.iqr_mean(iv), ...
            mom.cross_section.skew_mean(iv));
    end
end
fprintf('\n');

end


% Helper function : fetch re moments from dynare native path
function [V, AR1] = re_moments(dr, M_, options_)
nvar = M_.endo_nbr;
ivar = (1:nvar)';

opts = options_;
if ~isfield(opts, 'ar') || opts.ar < 1
    opts.ar = 1;                  
end

[Gamma_y, ~] = th_autocovariances(dr, ivar, M_, opts, 1);   
V   = Gamma_y{1};                 % [nvar x nvar]
AR1 = diag(Gamma_y{2});           % [nvar x 1] AR(1) autocorrelation
end
