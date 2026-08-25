function options_ = setup_options_AL(M_, options_, skip_simul_cheks)
% SETUP_OPTIONS_AL  Build options_.learning.adaptive from AL_ parameters in M_.
%
%   skip_simul_checks [logical, optional, 3rd arg, default false]: used
%   to distinguish if the function is called from the simulation path or estimation path
%   and run the right checks. Not usefull yet in adaptive learning as no
%   estimation has been build but kept to keep the same logic between all
%   learning type
%
% LEARNED VARIABLES (opt-in)
%   AL_learn_<var> = 1 marks the forward-looking variable <var> as learned.
%   AL_learn_<var> = 0 marks it as explicitly kept at rational expectations.
%   A forward variable with no AL_learn_<var> parameter is NOT learned.
%   Declaring no flag at all, or setting every flag to 0, is an error.
%
% AL_ parameter naming convention (all optional except AL_active):
%   AL_active         1 to enable AL (0/absent => return, standard RE)
%   AL_learn_<var>    1/0, selects the learned forward variables
%   AL_gain           constant-gain RLS step size                  [0.02]
%   AL_pj_threshold   projection-facility threshold on max|eig(T)| [1.0]
%   AL_plm_type       1 = AR(1) + constant (default) | 0 = intercept only
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

if nargin < 3 || isempty(skip_simul_checks)
    skip_simul_checks = false;
end

idx = find(strcmp(M_.param_names, 'AL_active'), 1);
al_active = ~isempty(idx) && isfinite(M_.params(idx)) && M_.params(idx) ~= 0;
if ~al_active && ~force_active
    return
end

learning.adaptive.check_support_AL(M_, options_);

A = struct();
A.AL_active    = true;
A.gain         = getp(M_, 'AL_gain', 0.02);
A.pj_threshold = getp(M_, 'AL_pj_threshold', 1.0);

% PLM type
plm_code = getp(M_, 'AL_plm_type', 1);
switch plm_code
    case 1, A.plm_type = 'ar1';       A.nb = 2;   % constant + slope
    case 0, A.plm_type = 'intercept'; A.nb = 1;   % constant only
    otherwise
        error('learning:adaptive:plm', ...
            'AL_plm_type=%g invalid (expected 1=ar1 or 0=intercept).', plm_code);
end

% forward variables
all_fwrd   = find(M_.lead_lag_incidence(3,:) > 0);
fwrd_names = M_.endo_names(all_fwrd);
nfwrd      = numel(all_fwrd);
if nfwrd == 0
    error('learning:adaptive:noFwrd', ...
        'No forward variable: nothing to learn under AL.');
end

%Learned var
pnames     = M_.param_names;
prefix_lrn = 'AL_learn_';

id_learn = false(nfwrd, 1);
any_flag = false;
for ip = 1:numel(pnames)
    pn = pnames{ip};
    if ~strncmp(pn, prefix_lrn, numel(prefix_lrn))
        continue;
    end
    vname = pn(numel(prefix_lrn)+1:end);
    iv    = find(strcmp(fwrd_names, vname), 1);
    if isempty(iv)
        if any(strcmp(M_.endo_names, vname))
            error('learning:adaptive:notForward', ...
                ['"%s" (from %s) is an endogenous variable but is not ' ...
                 'forward-looking: it cannot be learned.'], vname, pn);
        else
            error('learning:adaptive:unknownLearnVar', ...
                '"%s" (from %s) is not an endogenous variable.', vname, pn);
        end
    end
    any_flag     = true;
    id_learn(iv) = M_.params(ip) ~= 0;
end

if ~any_flag
    error('learning:adaptive:noLearnFlag', ...
        ['AL_active = 1 but no AL_learn_<var> parameter is declared, so no ' ...
         'variable would be learned. Declare AL_learn_<var> = 1 for each ' ...
         'learned forward variable (available: %s).'], ...
        strjoin(fwrd_names, ', '));
end
if ~any(id_learn)
    error('learning:adaptive:emptyLearnSet', ...
        ['All AL_learn_<var> flags are set to 0: no variable is learned. ' ...
         'Set AL_active = 0 to run the rational-expectations path.']);
end

A.id_fwrd     = all_fwrd(id_learn);
A.id_fwrd     = A.id_fwrd(:);
A.nfwrd       = numel(A.id_fwrd);
A.learn_names = M_.endo_names(A.id_fwrd);

% guards
if A.gain < 0 || A.gain > 1
    error('learning:adaptive:gain', 'AL_gain=%.4g out of [0,1].', A.gain);
end
if A.pj_threshold <= 0
    error('learning:adaptive:pj', 'AL_pj_threshold must be > 0.');
end

% Streams handling (not compatible with Octave)
has_good_algo = isfield(options_, 'DynareRandomStreams') && ...
    isfield(options_.DynareRandomStreams, 'algo') && ...
    ismember(options_.DynareRandomStreams.algo, {'mrg32k3a', 'mlfg6331_64'});
if ~has_good_algo
    prev_algo = 'mt19937ar';
    seed      = 0;
    if isfield(options_, 'DynareRandomStreams')
        if isfield(options_.DynareRandomStreams, 'algo'); prev_algo = options_.DynareRandomStreams.algo; end
        if isfield(options_.DynareRandomStreams, 'seed'); seed      = options_.DynareRandomStreams.seed; end
    end

    set_dynare_seed('mrg32k3a', seed);
    fprintf(['[learning] RNG "%s" has no substreams; promoted to mrg32k3a ' ...
        '(seed %d) so that learning streams can use substreams. ' ...
        'Declare set_dynare_seed(''mrg32k3a'', %d) in the .mod to make this ' ...
        'explicit and silence the notice.\n'], prev_algo, seed, seed);
end

%return option
options_.learning.adaptive = A;


% summary
if A.nb == 2
    plm_desc = 'constant + AR(1), equation-by-equation';
else
    plm_desc = 'intercept only (perceived level)';
end
if ~(isfield(options_,'noprint') && options_.noprint)
    fprintf('\n===== Adaptive Learning =====\n');
    fprintf('  PLM             : %s\n', plm_desc);
    fprintf('  constant gain   : %.4g\n', A.gain);
    fprintf('  projection thr. : %.4g\n', A.pj_threshold);
    fprintf('  forward vars    : %d\n', nfwrd);
    fprintf('  learned vars    : %s (%d)\n', strjoin(A.learn_names, ', '), A.nfwrd);
    re_fwrd = fwrd_names(~id_learn);
    if ~isempty(re_fwrd)
        fprintf('  kept at RE      : %s (%d)\n', strjoin(re_fwrd, ', '), numel(re_fwrd));
    end
    fprintf('=============================\n\n');
end
end

function v = getp(M_, name, default)
idx = find(strcmp(M_.param_names, name), 1);
if isempty(idx) || ~isfinite(M_.params(idx)); v = default; else; v = M_.params(idx); end
end
