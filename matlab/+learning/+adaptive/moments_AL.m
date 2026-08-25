function oo_ = moments_AL(M_, options_, oo_)
% MOMENTS_AL  Empirical moments of the model under Adaptive Learning
% (constant gain), by Monte Carlo over shock draws. 
%
% OPTIONS 
%   .periods  simulation length per realization
%   .drop     burn-in discarded before measuring
%   .replic   number of MC realizations
%   .ar       number of autocorrelation lags        [5]
%   .noprint  suppress display
%
% OUTPUTS (stored in oo_)
%   oo_.mean [nvar x 1], oo_.var [nvar x nvar], oo_.autocorr{1:ar}
%   oo_.learning.adaptive.moments.{mean_ci, std_ci, autocorr_ci, beliefs,
%                                  pf_hits, mc_raw, config}
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

disp('MC simulation for empirical AL moments running')

periods  = options_.periods;
drop     = options_.drop;
replic   = options_.replic;
T_measure = periods - drop;

if T_measure < 100
    error('learning:adaptive:shortSample', ...
          ['T_measure = %d too short (periods=%d, drop=%d). Increase periods ' ...
           'or decrease drop.'], T_measure, periods, drop);
end
if replic < 2
    warning('learning:adaptive:singleReplic', ...
        ['replic = %d: AL moment confidence intervals require replic > 1. ' ...
         'Set replic in the stoch_simul command.'], replic);
end

ar_order = 5;
if isfield(options_, 'ar'); ar_order = options_.ar; end
noprint = isfield(options_, 'noprint') && options_.noprint;

A     = options_.learning.adaptive;
nvar  = M_.endo_nbr;
nexo  = M_.exo_nbr;
nfwrd = A.nfwrd;
nb    = A.nb;

id_struct   = find(diag(M_.Sigma_e) > 0);
cs          = get_lower_cholesky_covariance(M_.Sigma_e, options_.add_tiny_number_to_cholesky);
chol_struct = cs(id_struct, id_struct)';


mc_mean     = zeros(replic, nvar);
mc_var      = zeros(replic, nvar, nvar);
mc_autocorr = zeros(replic, nvar, nvar, ar_order);
mc_bel_mean = zeros(replic, nfwrd, nb);       % time-average of belief coeffs
mc_bel_std  = zeros(replic, nfwrd, nb);       % time-volatility of belief coeffs
mc_pf       = zeros(replic, 1);

if ~noprint
    fprintf('\n  AL moments: %d MC realizations, drop=%d, T_measure=%d\n', ...
            replic, drop, T_measure);
end

t_start = tic;
base_seed = options_.DynareRandomStreams.seed;
algo      = options_.DynareRandomStreams.algo;
[belief_rng,~] = learning.learning_streams(base_seed, algo, 'sim', 1);

parfor k = 1:replic
    % Beliefs are deterministic under AL: only the shock substream is used
    [~, shock_k] = learning.learning_streams(base_seed, algo, 'sim', k);

    ee_k = zeros(periods, nexo);
    ee_k(:, id_struct) = randn(shock_k, periods, numel(id_struct)) * chol_struct;
    oo_k = learning.adaptive.simult_AL(M_, options_, oo_, ee_k, belief_rng);

    info_k= oo_k.learning.adaptive.info;
    y_k= oo_k.endo_simul ;

    idx_m = (drop + 1):periods;                % T_measure periods
    y_m   = y_k(:, idx_m)';                     % (T_measure x nvar)

    mc_mean(k, :)   = mean(y_m, 1);
    mc_var(k, :, :) = cov(y_m);

    ac_k = zeros(nvar, nvar, ar_order);
    for lag = 1:ar_order
        for iv = 1:nvar
            for jv = 1:nvar
                cc = corrcoef(y_m(1+lag:end, iv), y_m(1:end-lag, jv));
                ac_k(iv, jv, lag) = cc(1,2);
            end
        end
    end
    mc_autocorr(k, :, :, :) = ac_k;

    % Belief-coefficient dynamics
    bel_m = info_k.beta_hist(:, :, idx_m);      % [nfwrd x nb x T_measure]
    mc_bel_mean(k, :, :) = mean(bel_m, 3);
    mc_bel_std(k, :, :)  = std(bel_m, 0, 3);
    mc_pf(k) = info_k.pf_hits;
end

elapsed = toc(t_start);

% aggregation
oo_.mean = mean(mc_mean, 1)';
oo_.var  = squeeze(mean(mc_var, 1));
oo_.autocorr = cell(ar_order, 1);
for lag = 1:ar_order
    oo_.autocorr{lag} = squeeze(mean(mc_autocorr(:,:,:,lag), 1));
end

moments = struct();
moments.mean_ci = [quantile(mc_mean, 0.025, 1); quantile(mc_mean, 0.975, 1)]';

moments.std_ci = zeros(nvar, 2);
for iv = 1:nvar
    std_k = sqrt(mc_var(:, iv, iv));
    moments.std_ci(iv, :) = [quantile(std_k, 0.025), quantile(std_k, 0.975)];
end

moments.autocorr_ci = cell(ar_order, 1);
for lag = 1:ar_order
    ac_diag = zeros(replic, nvar);
    for iv = 1:nvar
        ac_diag(:, iv) = mc_autocorr(:, iv, iv, lag);
    end
    moments.autocorr_ci{lag} = [quantile(ac_diag, 0.025, 1); ...
                                 quantile(ac_diag, 0.975, 1)]';
end

% Belief dynamics
cm = mean(mc_bel_mean, 1);  csd = mean(mc_bel_std, 1);
moments.beliefs.coef_mean = reshape(cm, nfwrd, nb);
moments.beliefs.coef_std  = reshape(csd, nfwrd, nb);

moments.pf_hits = mc_pf;

moments.mc_raw.mean     = mc_mean;
moments.mc_raw.var      = mc_var;
moments.mc_raw.autocorr = mc_autocorr;
moments.mc_raw.bel_mean = mc_bel_mean;
moments.mc_raw.bel_std  = mc_bel_std;

moments.config = struct('periods',periods,'drop',drop,'replic',replic, ...
                        'ar',ar_order,'gain',A.gain,'plm_type',A.plm_type);

oo_.learning.adaptive.moments = moments;

if ~noprint
    fprintf('  Done in %.1f s (%.2f s/real.)\n\n', elapsed, elapsed/replic);
    learning.adaptive.disp_moments_AL(M_, options_, oo_);
end

end