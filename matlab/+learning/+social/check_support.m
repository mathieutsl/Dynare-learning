function check_support(M_, options_, estim_params_, context)
% CHECK_SUPPORT  Reject or flag configurations the SL toolbox does not handle.
%
% INPUTS
%   M_, options_    standard Dynare structures
%   estim_params_   [] on the simulation path
%   context         'simul' or 'estim' shared helpers
%
% Options not listed here are untested and should be used carefully 
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

% common to estim and simul
if options_.order > 1
    error('learning:social:order', ...
        ['Social learning requires a first-order approximation (order = %d). ' ...
        'The expectations matrix R is derived from the first-order solution.'], ...
        options_.order);
end

if options_.loglinear
    error('learning:social:loglinear', ...
        ['Social learning is not compatible with the loglinear option: R is ' ...
        'computed from the dynamic Jacobian evaluated at oo_.dr.ys, which ' ...
        'loglinear replaces by its logarithm.']);
end

if M_.exo_det_nbr > 0
    error('learning:social:exoDet', ...
        'Deterministic exogenous variables are not supported (got %d).', M_.exo_det_nbr);
end

switch context
    case 'simul'
        if isfield(M_,'endo_histval') && ~isempty(M_.endo_histval)
            error('learning:social:histval', ...
                ['histval is ignored under social learning: simult_SL always starts ' ...
                'from the steady state.']);
        end
        hp = learning.getopt(options_, 'hp_filter', 0);
        bp = 0;
        if isfield(options_, 'bandpass') && isstruct(options_.bandpass)
            bp = learning.getopt(options_.bandpass, 'indicator', 0);
        end
        if hp || bp
            warning('learning:social:filteredMoments', ...
                ['SL moments are computed on unfiltered simulated data while the RE ' ...
                'benchmark goes through th_autocovariances, which honours the filter ' ...
                'options. The two columns of the comparison table are not comparable.']);
        end

    case 'estim'
        nvn = size(learning.getopt(estim_params_, 'var_endo', []), 1);   % measurement error std
        ncn = size(learning.getopt(estim_params_, 'corrn',    []), 1);   % ME correlations
        ncx = size(learning.getopt(estim_params_, 'corrx',    []), 1);   % shock correlations

        if nvn + ncn > 0
            error('learning:social:measurementError', ...
                ['Measurement errors are not supported: the inversion filter requires ' ...
                'exactly as many structural shocks as observables.']);
        end
        if ncx > 0
            error('learning:social:corrShocks', ...
                ['Estimated correlations between structural shocks are not supported: ' ...
                'the Ox/Or split assumes news and structural shocks are orthogonal.']);
        end
        if size(learning.getopt(estim_params_, 'skew_exo', []), 1) > 0
            error('learning:social:skewedShocks', ...
                ['Skewed structural shocks are not supported: the inversion filter ' ...
                'assumes Gaussian innovations.']);
        end
        if isfield(M_,'H') && ~isempty(M_.H) && any(M_.H(:) ~= 0)
            error('learning:social:measurementError', ...
                'Calibrated measurement errors (M_.H) are not supported.');
        end
        if learning.getopt(options_,'diffuse_filter',false)
            error('learning:social:diffuseFilter', ...
                'The diffuse filter is not supported by the SL inversion filter.');
        end
        if learning.getopt(options_,'heteroskedastic_filter',false)
            error('learning:social:heteroskedastic', ...
                'Heteroskedastic shocks are not supported: Sigma_e is assumed constant.');
        end
        if ~isempty(learning.getopt(options_,'trend_coeffs',[]))
            error('learning:social:trends', ...
                'Deterministic trends in the measurement equation are not supported.');
        end
        if learning.getopt(options_,'forecast',0) > 0
            error('learning:social:forecast', ...
                ['Forecasts are not supported: store_SL_smoother does not populate ' ...
                'the oo_.Smoother fields required by forecasts.run.']);
        end
        % requested outputs the SL smoother does not produce
        unsupported = {'filtered_vars','filter_covariance','filter_decomposition', ...
            'moments_varendo','bayesian_irf'};
        flagged = unsupported(cellfun(@(o) logical(learning.getopt(options_,o,false)), unsupported));
        if ~isempty(flagged)
            warning('learning:social:unsupportedOutput', ...
                ['The following options are ignored under social learning: %s. ' ...
                'They rely on Kalman-filter objects the inversion filter does not ' ...
                'produce.'], strjoin(flagged, ', '));
        end

    otherwise
        error('learning:social:badContext', 'context must be ''simul'' or ''estim''.');
end
end