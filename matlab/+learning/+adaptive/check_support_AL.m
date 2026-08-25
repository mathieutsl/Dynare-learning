function check_support_AL(M_, options_)
% CHECK_SUPPORT_AL  Reject or flag configurations the AL simulation does not
% handle. Simulation-only POC: cases where silence would give wrong results
% raise an error; cases where an output is simply not produced raise a warning.
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

if options_.order > 1
    error('learning:adaptive:order', ...
        ['Adaptive learning requires a first-order approximation (order = %d). ' ...
        'The ALM re-solve is linear.'], options_.order);
end

if options_.loglinear
    error('learning:adaptive:loglinear', ...
        ['Adaptive learning is not compatible with the loglinear option: the ' ...
        'Jacobian is evaluated at oo_.dr.ys, which loglinear replaces by its ' ...
        'logarithm.']);
end

if M_.exo_det_nbr > 0
    error('learning:adaptive:exoDet', ...
        'Deterministic exogenous variables are not supported (got %d).', M_.exo_det_nbr);
end

if isfield(M_,'endo_histval') && ~isempty(M_.endo_histval)
    error('learning:adaptive:histval', ...
        ['histval is ignored under adaptive learning: simult_AL always starts ' ...
        'from the steady state.']);
end

hp = learning.getopt(options_, 'hp_filter', 0);
bp = 0;
if isfield(options_, 'bandpass') && isstruct(options_.bandpass)
    bp = learning.getopt(options_.bandpass, 'indicator', 0);
end
if hp || bp
    warning('learning:adaptive:filteredMoments', ...
        ['AL moments are computed on unfiltered simulated data while the RE ' ...
        'benchmark goes through th_autocovariances, which honours the filter ' ...
        'options. The two columns of the comparison table are not comparable.']);
end
end