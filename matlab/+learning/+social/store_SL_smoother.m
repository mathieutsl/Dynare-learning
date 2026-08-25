function oo_ = store_SL_smoother(oo_, M_, options_, model_solution)
%STORE_SL_SMOOTHER  Route the IF_SL smoother payload into oo_
%   oo_ = store_SL_smoother(oo_, M_, options_, model_solution)
%
%   INPUTS
%     oo_             results structure.
%     M_              model structure.
%     options_        must contain a populated options_.learning.social
%     model_solution  the 6th output of dsge_likelihood_SL, evaluated at the
%                     mode/mean
%
%   OUTPUTS
%     oo_.SmoothedShocks.<exo>            smoothed structural shocks.
%     oo_.SmoothedVariables.<endo>        smoothed state trajectory.
%     oo_.Smoother.SteadyState            steady state.
%     oo_.Smoother.Constant.<endo>        constant removed from the measurement
%                                         equation.
%     oo_.learning.social.smoother        aggregate expectations, realized forward
%                                         states and news shock series.
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

L         = options_.learning.social;
ee        = model_solution.learning.social.ee;         % T x N 
y_        = model_solution.learning.social.y_;         
stop_dist = model_solution.learning.social.stop_dist;  
expectation = model_solution.learning.social.expectation;        % nfwrd x T
ys        = model_solution.dr.ys;

T = size(ee, 1);

% column alignment
ncol   = size(y_, 2);
navail = ncol - 1;
nfill  = min([T - stop_dist, navail, T]);
if stop_dist > 0
    warning('learning:social:earlyStop', ...
        ['store_SL_smoother: the inversion stopped %d periods before the end ' ...
        'of the sample; only %d of %d periods are smoothed, the rest is NaN.'], ...
        stop_dist, nfill, T);
end

% blank out the periods the inversion never reached
if nfill < T
    ee(nfill+1:end, :)  = NaN;
    expectation(:, nfill+1:end) = NaN;
end

% smoothed shocks
for j = 1:M_.exo_nbr
    oo_.SmoothedShocks.(M_.exo_names{j}) = ee(:, j);
end

% smoothed variables
Ysm = nan(M_.endo_nbr, T);
Ysm(:, 1:nfill) = y_(:, 2:1+nfill);
for iv = 1:M_.endo_nbr
    oo_.SmoothedVariables.(M_.endo_names{iv}) = Ysm(iv, :)';
end

% SL smoother payload
news_cols = find(any(L.Or ~= 0, 1));

smoother = struct();
smoother.forward_vars   = M_.endo_names(L.id_fwrd);

% aggregate expectation entering the policy function (belief + news)
smoother.expectations   = expectation';                    % T x nfwrd
smoother.forward_states = Ysm(L.id_fwrd, :)';      % T x nfwrd
smoother.news_shocks    = M_.exo_names(news_cols);

for c = news_cols(:)'
    smoother.news.(M_.exo_names{c}) = ee(:, c);
end
smoother.done = true;

oo_.learning.social.smoother = smoother;

if isfield(options_,'prefilter') && options_.prefilter
    cst_val = zeros(M_.endo_nbr, 1);
else
    cst_val = ys;
end
for iv = 1:M_.endo_nbr
    oo_.Smoother.Constant.(M_.endo_names{iv}) = cst_val(iv);
end
oo_.Smoother.SteadyState = ys;
oo_.Smoother.loglinear   = false;

if ~(isfield(options_,'noprint') && options_.noprint)
    fprintf('store_SL_smoother: wrote %d smoothed shocks, %d smoothed variables (%d/%d periods).\n', ...
        M_.exo_nbr, M_.endo_nbr, nfill, T);
end
end