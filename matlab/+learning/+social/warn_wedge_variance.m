function warn_wedge_variance(M_)
% WARN_WEDGE_VARIANCE  Diagnose belief-wedge shocks left with a non-zero
%   variance while social learning is switched off.
%
%   warn_wedge_variance(M_) scans the exogenous variables for the a_<var>
%   naming convention used by the social-learning package, where <var> is a
%   forward-looking endogenous variable, and warns for each one whose
%   variance is strictly positive.
%
%
% INPUTS
%   M_  [struct]  Dynare model structure
%
% OUTPUTS
%   none (raises warning learning:social:wedgeVariance)
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

if ~isfield(M_, 'Sigma_e') || isempty(M_.Sigma_e)
    return
end
if ~isfield(M_, 'lead_lag_incidence') || isempty(M_.lead_lag_incidence)
    return
end

id_fwrd = find(M_.lead_lag_incidence(3, :) > 0);
fwrd    = M_.endo_names(id_fwrd);

for i = 1:numel(fwrd)
    f  = fwrd{i};
    sk = ['a_' f];
    k  = find(strcmp(M_.exo_names, sk), 1);
    if isempty(k)
        continue
    end
    v = M_.Sigma_e(k, k);
    if v > 0
        warning('learning:social:wedgeVariance', ...
            ['Social learning is OFF but wedge shock %s carries a non-zero ' ...
            'variance (var = %.4g). %s adds a wedge to the expectation of ' ...
            '%s (rational value + %s). Two mutually exclusive readings -- ' ...
            'pick one:\n' ...
            '  (a) set var %s = 0  -> pure rational expectations for %s.\n' ...
            '  (b) keep it non-zero -> %s is an ordinary structural ' ...
            'expectation shock, and this run is "RE + expectation shock", ' ...
            'NOT pure RE.\n' ...
            'The silent default is (b): it runs and yields plausible but ' ...
            'non-RE results.'], ...
            sk, v, sk, f, sk, sk, f, sk);
    end
end
end