function oo_ = simult_AL(M_, options_, oo_, ee, ~)
% SIMULT_AL  Simulate the model under Adaptive Learning (constant gain).
%
% REFERENCES
%   Slobodyan, S. and R. Wouters (2012), "Learning in a Medium-Scale DSGE Model
%   with Expectations Based on Small Forecasting Models", American Economic
%   Journal: Macroeconomics, 4(2), 65-101.
%
%   The recursive least squares updating, the projection facility and the
%   structure of the belief/ALM split follow the algebra of the replication
%   code accompanying that paper. No code was copied verbatim.
%
% Each period: beliefs are updated by constant-gain RLS (update_beliefs_AL),
% then the ALM is re-solved from the current beliefs (resolve_alm_AL). The
% projection facility keeps the last stable ALM if an update destabilises it.
%
% INPUTS
%   M_, options_, oo_   standard Dynare structures; options_.learning.adaptive
%                       must have been built by setup_options_AL.
%   ee     [T x nexo]   exogenous shock realizations, same layout as simult_SL.
%   ~                   RNG placeholder: constant-gain AL is deterministic, the
%                       argument is accepted so that both mechanisms share one
%                       entry-point signature.
%
% OUTPUTS
%   oo_.endo_simul                        [nvar x T]
%   oo_.learning.adaptive.info.beta_final  [nfwrd x nb]      terminal beliefs
%   oo_.learning.adaptive.info.beta_hist   [nfwrd x nb x T]  belief path
%   oo_.learning.adaptive.info.pf_hits     [scalar]          projection facility hits
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

A  = options_.learning.adaptive;
n  = M_.endo_nbr; id = A.id_fwrd; nf = A.nfwrd;

T_sim = size(ee, 1);
if size(ee, 2) ~= M_.exo_nbr
    error('learning:adaptive:badShockSize', ...
        'ee must be [T x %d], got [%d x %d].', M_.exo_nbr, size(ee,1), size(ee,2));
end
if T_sim < 1
    error('learning:adaptive:emptySample', 'ee is empty.');
end

J = local_jac_blocks(M_, oo_);
[beta, Rmom] = learning.adaptive.init_beliefs_AL(M_, options_,oo_);
[T, R, mu, ok] = learning.adaptive.resolve_alm_AL(beta, J, M_, options_);
if ~ok
    error('learning:adaptive:unstableInitialALM', 'Initial ALM is unstable.');
end

y         = zeros(n, T_sim+1);        % col 1 = SS (deviations = 0)
yf_lag    = zeros(nf, 1);
beta_hist = zeros(nf, A.nb, T_sim);
pf_hits   = 0;

for t = 1:T_sim
    y(:,t+1) = mu + T*y(:,t) + R*ee(t,:)';
    yf_now   = y(id, t+1);
    if A.nb == 2
        z_lag = [ones(1,nf); yf_lag'];   % ar1: [1; lagged forward]
    else
        z_lag = ones(1,nf);              % intercept: constant only
    end

    [beta_new, Rmom_new] = learning.adaptive.update_beliefs_AL( ...
        beta, Rmom, yf_now, z_lag, A.gain);
    [T2, R2, mu2, ok] = learning.adaptive.resolve_alm_AL(beta_new, J, M_, options_);
    if ok
        beta = beta_new;  Rmom = Rmom_new;
        T = T2;  R = R2;  mu = mu2;
    else
        pf_hits = pf_hits + 1;
    end
    beta_hist(:,:,t) = beta;
    yf_lag = yf_now;
end

oo_.endo_simul = y(:, 2:end);            % drop the SS column

info.beta_final = beta;
info.beta_hist  = beta_hist;
info.pf_hits    = pf_hits;

oo_.learning.adaptive.info = info;
end


% Helper
function J = local_jac_blocks(M_, oo_)
n = M_.endo_nbr; ne = M_.exo_nbr;
myss = repmat(oo_.dr.ys, 3, 1);
[Jac,~,~] = feval([M_.fname '.dynamic_g1'], myss, zeros(1,ne), M_.params, oo_.dr.ys, ...
    M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, M_.dynamic_g1_sparse_colptr);
if issparse(Jac); Jac = full(Jac); end
J.HH = Jac(:, 1:n);
J.GG = Jac(:, n+1:2*n);
J.FF = Jac(:, 2*n+1:3*n);
J.MM = Jac(:, 3*n+1:3*n+ne);
if ~all(isfinite([J.HH(:);J.GG(:);J.FF(:);J.MM(:)]))
    error('learning:adaptive:nonfiniteJacobian', 'Non-finite Jacobian blocks.');
end
end