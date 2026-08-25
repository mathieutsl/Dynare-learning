function oo_ = moments_SL(M_, options_, oo_)
% MOMENTS_SL  Empirical moments of the model under Social Learning, by Monte
% Carlo over agent draws.
%
% OPTIONS  
%   .periods   simulation length per realization    
%   .drop      burn-in periods discarded            
%   .replic    number of MC realizations            
%   .ar        number of autocorrelation lags       
%   .noprint   suppress display                       
%
%
% BELIEF HYPERPARAMETERS  (options_.learning.social, from setup_options_SL)
%   .N_agents, .mut_p, .mut_sd, .rhoGN, .wed_p, .t_max, .Or, .Ox
%
% OUTPUTS  (stored in oo_)
%   oo_.mean [nvar x 1], oo_.var [nvar x nvar], oo_.autocorr{1:ar}
%   oo_.learning.social.moments.{mean_ci, std_ci, autocorr_ci, cross_section, mc_raw, config}
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

disp('MC Simulation for empirical SL moments running')

periods  = options_.periods;     % length per realization
drop     = options_.drop;        % burn-in discarded before measuring
replic   = options_.replic;      % number of MC realizations
T_measure = periods - drop;

if T_measure < 100
    error('learning:social:shortSample', ...
          ['T_measure = %d too short (periods=%d, drop=%d). Increase periods ' ...
           'or decrease drop.'], T_measure, periods, drop);
end
if replic < 2
    warning('learning:social:singleReplic', ...
        ['replic = %d: SL moment confidence intervals and belief dispersion ' ...
         'require replic > 1. Set replic in the stoch_simul command.'], replic);
end

ar_order = 5;
if isfield(options_, 'ar'); ar_order = options_.ar; end
noprint = isfield(options_, 'noprint') && options_.noprint;

if ~isfield(oo_.dr, 'ghr') || isempty(oo_.dr.ghr)
    oo_ = learning.social.compute_ghr(M_, options_, oo_);
end

nvar    = M_.endo_nbr;
nexo    = M_.exo_nbr;
nfwrd   = sum(M_.lead_lag_incidence(3,:) > 0);

id_struct   = find(diag(M_.Sigma_e) > 0);
cs          = get_lower_cholesky_covariance(M_.Sigma_e, options_.add_tiny_number_to_cholesky);
chol_struct = cs(id_struct, id_struct)';

base_seed = options_.DynareRandomStreams.seed;
algo      = options_.DynareRandomStreams.algo;

mc_mean     = zeros(replic, nvar);
mc_var      = zeros(replic, nvar, nvar);
mc_autocorr = zeros(replic, nvar, nvar, ar_order);

mc_cs_std   = zeros(replic, nfwrd, T_measure);
mc_cs_iqr   = zeros(replic, nfwrd, T_measure);
mc_cs_skew  = zeros(replic, nfwrd, T_measure);

if ~noprint
    fprintf('\n  SL Moments: %d MC replications, drop=%d, T_measure=%d\n', ...
        replic, drop, T_measure);
end

t_start = tic;

parfor k = 1:replic

    % Two independent substreams of one seed: beliefs (2k-1), shocks (2k).
    [belief_k, shock_k] = learning.learning_streams(base_seed, algo, 'sim', k);

    ee_k = zeros(periods, nexo);
    ee_k(:, id_struct) = randn(shock_k, periods, numel(id_struct)) * chol_struct;
    oo_k = learning.social.simult_SL(M_, options_, oo_, ee_k, belief_k);
    

    idx_m = (drop + 1):periods;                 % T_measure periods
    y_m   = oo_k.endo_simul(:, idx_m)';          % (T_measure x nvar)

    % Aggregate moments
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

    % Cross-section moments of beliefs, same window
    bel_m = oo_k.learning.social.beliefs(:, :, idx_m);
    cs_std_k  = zeros(nfwrd, T_measure);
    cs_iqr_k  = zeros(nfwrd, T_measure);
    cs_skew_k = zeros(nfwrd, T_measure);
    for t = 1:T_measure
        for iv = 1:nfwrd
            bel_t = bel_m(iv, :, t);
            sd_t  = std(bel_t);
            cs_std_k(iv, t) = sd_t;
            cs_iqr_k(iv, t) = iqr(bel_t);
            if sd_t > 1e-15
                cs_skew_k(iv, t) = skewness(bel_t);
            end
        end
    end
    mc_cs_std(k, :, :)  = cs_std_k;
    mc_cs_iqr(k, :, :)  = cs_iqr_k;
    mc_cs_skew(k, :, :) = cs_skew_k;

end

elapsed = toc(t_start);

% aggregation 
oo_.mean = mean(mc_mean, 1)';
oo_.var  = squeeze(mean(mc_var, 1));

oo_.autocorr = cell(ar_order, 1);
for lag = 1:ar_order
    oo_.autocorr{lag} = squeeze(mean(mc_autocorr(:,:,:,lag), 1));
end

% Confidence intervals
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

% Cross-section
moments.cross_section.std_mean  = squeeze(mean(mean(mc_cs_std, 3), 1));
moments.cross_section.iqr_mean  = squeeze(mean(mean(mc_cs_iqr, 3), 1));
moments.cross_section.skew_mean = squeeze(mean(mean(mc_cs_skew, 3), 1));
moments.cross_section.std_ts    = squeeze(mean(mc_cs_std, 1));
moments.cross_section.iqr_ts    = squeeze(mean(mc_cs_iqr, 1));
moments.cross_section.skew_ts   = squeeze(mean(mc_cs_skew, 1));

% Raw MC data
moments.mc_raw.mean     = mc_mean;
moments.mc_raw.var      = mc_var;
moments.mc_raw.autocorr = mc_autocorr;
moments.mc_raw.cs_std   = mc_cs_std;
moments.mc_raw.cs_iqr   = mc_cs_iqr;
moments.mc_raw.cs_skew  = mc_cs_skew;


oo_.learning.social.moments = moments;

% Display
if ~noprint
    fprintf('  Completed in %.1f s (%.2f s/rep.)\n\n', elapsed, elapsed/replic);
    learning.social.disp_moments_SL(M_, options_, oo_);
end

end
