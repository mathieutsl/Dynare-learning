function [y, oo_] = irf_SL(M_, options_, oo_, shock_idx)
% IRF_SL  Generalized IRF under Social Learning.
%
% OPTIONS 
%   options_.irf     horizon H
%   options_.replic  number of MC replications
%   options_.drop    common burn-in before the impulse 
%
% BANDS
%   options_.learning.social.irf_quantiles  [.05 .5 .95]
%
% OUTPUTS
%   y   [nvar x H]  median GIRF
%   oo_.learning.social.irf.<shock>.{median, mean, bands}
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
nexo = M_.exo_nbr;

H       = options_.irf;
replic  = options_.replic;
drop    = options_.drop;
sname   = strtrim(M_.exo_names{shock_idx});

qs = learning.getopt(options_.learning.social, 'irf_quantiles', [0.05 0.5 0.95]);

if ~isfield(oo_.dr,'ghr') || isempty(oo_.dr.ghr)
    oo_ = learning.social.compute_ghr(M_, options_, oo_);
end

base_seed = options_.DynareRandomStreams.seed;
algo      = options_.DynareRandomStreams.algo;

cs = get_lower_cholesky_covariance(M_.Sigma_e, options_.add_tiny_number_to_cholesky);
id_struct = find(diag(M_.Sigma_e) > 0);
chol_struct = cs(id_struct, id_struct)';

girf = zeros(nvar, H, replic);

parfor rep = 1:replic

    % beliefs on substream 2*rep-1, shocks on 2*rep 
    [belief_rep, shock_rep] = learning.learning_streams(base_seed, algo, 'sim', rep);

    % common burn-in shocks
    ee_burn = zeros(drop, nexo);
    ee_burn(:, id_struct) = randn(shock_rep, drop, numel(id_struct)) * chol_struct;

    ee_b = [ee_burn; zeros(H, nexo)];
    ee_s = ee_b;
    ee_s(drop + 1, :) = cs(:, shock_idx)';      % orthogonalised impulse

    snap  = belief_rep.State;
    tmp_b = learning.social.simult_SL(M_, options_, oo_, ee_b, belief_rep);
  
    belief_rep.State = snap;
    tmp_s = learning.social.simult_SL(M_, options_, oo_, ee_s, belief_rep);
 
    idx_irf = (drop + 1):(drop + H);
    girf(:, :, rep) = tmp_s.endo_simul(:, idx_irf) - tmp_b.endo_simul(:, idx_irf);
end

% aggregation
irf_median = median(girf, 3);
irf_mean   = mean(girf, 3);
irf_bands  = quantile(girf, qs, 3);

y = irf_median;

oo_.learning.social.irf.(sname).median = irf_median;
oo_.learning.social.irf.(sname).mean   = irf_mean;
oo_.learning.social.irf.(sname).bands  = irf_bands;

end
