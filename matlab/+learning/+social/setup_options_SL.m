function options_ = setup_options_SL(M_, options_,skip_simul_checks)
% SETUP_OPTIONS_SL  Build options_.learning.social from SL_ parameters in M_.
%
%   skip_simul_checks [logical, optional, 3rd arg, default false]: used
%   to distinguish if the function is called from the simulation path or estimation path
%   and run the right checks
%
% LEARNED VARIABLES
%   SL_learn_<var> = 1 marks the forward-looking variable <var> as learned.
%   SL_learn_<var> = 0 marks it as explicitly kept at rational expectations.
%   A forward variable with no SL_learn_<var> parameter is not learned.
%
% SL_ parameter naming convention :
%   SL_active                 -> activation switch
%   SL_learn_<var>            -> 1/0, selects the learned forward variables
%   SL_n_agents               -> options_.learning.social.N_agents  (J)
%   SL_mutation_prob          -> options_.learning.social.mut_p     (mu, global default)
%   SL_wedding_proportion     -> options_.learning.social.wed_p
%   SL_t_max                  -> options_.learning.social.t_max     (expectation-ahead horizons)
%   SL_mutation_std_<var>     -> options_.learning.social.mut_sd for <var>  (xi)   [required if learned]
%   SL_fitness_discount_<var> -> options_.learning.social.rhoGN  for <var>  (delta) [required if learned]
%   SL_mutation_prob_<var>    -> options_.learning.social.mut_p  for <var>  (mu), per-var override
%   Shock convention: a_<var> in varexo, required for every learned variable
%
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

idx = find(strcmp(M_.param_names, 'SL_active'), 1);
if isempty(idx) || ~isfinite(M_.params(idx)) || M_.params(idx) == 0
    error('learning:social:notActive', ...
        ['setup_options_SL was called with SL_active absent or 0. The ' ...
         'caller is responsible for the activation test; see the dispatch ' ...
         'in stoch_simul and dynare_estimation_1.']);
end

id_fwrd    = find(M_.lead_lag_incidence(3, :) > 0);
nfwrd      = numel(id_fwrd);
fwrd_names = M_.endo_names(id_fwrd);

if nfwrd == 0
    error('learning:social:noFwrd', ...
        'No forward-looking variable: nothing to learn under social learning.');
end

L = struct();
L.type = 'social';

%guard
if ~skip_simul_checks
    learning.social.check_support(M_, options_, [], 'simul');
end

L.N_agents = getp(M_, 'SL_n_agents',           300);
L.wed_p    = getp(M_, 'SL_wedding_proportion',  1.0);
L.t_max    = getp(M_, 'SL_t_max',                 1);

if mod(L.N_agents, 2) ~= 0
    warning('learning:social:oddAgents', ...
        'The number of agents (%d) is odd, which is not supported for tournament selection. It has been rounded up to %d.', ...
        L.N_agents, L.N_agents + 1);
    L.N_agents = L.N_agents + 1;
end


pnames     = M_.param_names;
prefix_lrn = 'SL_learn_';
prefix_mut = 'SL_mutation_std_';
prefix_dis = 'SL_fitness_discount_';
prefix_prb = 'SL_mutation_prob_';

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
            error('learning:social:notForward', ...
                ['"%s" (from %s) is an endogenous variable but is not ' ...
                 'forward-looking: it cannot be learned.'], vname, pn);
        else
            error('learning:social:unknownLearnVar', ...
                '"%s" (from %s) is not an endogenous variable.', vname, pn);
        end
    end
    any_flag     = true;
    id_learn(iv) = M_.params(ip) ~= 0;
end

if ~any_flag
    error('learning:social:noLearnFlag', ...
        ['SL_active = 1 but no SL_learn_<var> parameter is declared, so no ' ...
         'variable would be learned. Declare SL_learn_<var> = 1 for each ' ...
         'learned forward variable (available: %s).'], ...
        strjoin(fwrd_names, ', '));
end
if ~any(id_learn)
    error('learning:social:emptyLearnSet', ...
        ['All SL_learn_<var> flags are set to 0: no variable is learned. ' ...
         'Set SL_active = 0 to run the rational-expectations path.']);
end

% Per variable
L.mut_p  = zeros(nfwrd, 1);
L.mut_sd = zeros(nfwrd, 1);
L.rhoGN  = zeros(nfwrd, 1);
Or       = zeros(nfwrd, M_.exo_nbr);

for k = 1:nfwrd
    if ~id_learn(k)
        continue;
    end
    vname = fwrd_names{k};

    % xi: required
    idx_mut = find(strcmp(pnames, [prefix_mut vname]), 1);
    if isempty(idx_mut)
        error('learning:social:missingMutStd', ...
            ['"%s" is flagged learned (SL_learn_%s = 1) but ' ...
             'SL_mutation_std_%s is not declared.'], vname, vname, vname);
    end
    L.mut_sd(k) = M_.params(idx_mut);

    % delta: required
    idx_disc = find(strcmp(pnames, [prefix_dis vname]), 1);
    if isempty(idx_disc)
        error('learning:social:missingDiscount', ...
            ['"%s" is flagged learned (SL_learn_%s = 1) but ' ...
             'SL_fitness_discount_%s is not declared.'], vname, vname, vname);
    end
    L.rhoGN(k) = M_.params(idx_disc);

    % mu: required
    idx_prob = find(strcmp(pnames, [prefix_prb vname]), 1);
    if isempty(idx_prob)
        error('learning:social:missingMutProb', ...
            ['"%s" is flagged learned (SL_learn_%s = 1) but ' ...
            'SL_mutation_prob_%s is not declared.'], vname, vname, vname);
    end
    L.mut_p(k) = M_.params(idx_prob);

    % news shock, convention a_<var>: required
    is = find(strcmp(M_.exo_names, ['a_' vname]), 1);
    if isempty(is)
        error('learning:social:noShock', ...
            ['"%s" is flagged learned (SL_learn_%s = 1) but the news shock ' ...
             '"a_%s" was not found in varexo.'], vname, vname, vname);
    end
    Or(k, is) = 1;
end

for ip = 1:numel(pnames)
    pn = pnames{ip};
    vname = '';
    if strncmp(pn, prefix_mut, numel(prefix_mut))
        vname = pn(numel(prefix_mut)+1:end);
    elseif strncmp(pn, prefix_dis, numel(prefix_dis))
        vname = pn(numel(prefix_dis)+1:end);
    elseif strncmp(pn, prefix_prb, numel(prefix_prb))
        vname = pn(numel(prefix_prb)+1:end);
    end
    if isempty(vname)
        continue;
    end
    iv = find(strcmp(fwrd_names, vname), 1);
    if ~isempty(iv) && ~id_learn(iv)
        warning('learning:social:orphanParam', ...
            ['%s is declared but "%s" is not learned (SL_learn_%s is absent ' ...
             'or 0), so this parameter has no effect.'], pn, vname, vname);
    end
end

L.Or = Or;
L.Ox = eye(M_.exo_nbr) - Or' * Or;

if learning.is_active(options_, 'social')
    L.decomp = learning.getopt(options_.learning.social, 'decomp', false);
end

L.id_SL = find(id_learn);
L.n_SL  = numel(L.id_SL);

% Controls
bad = L.id_SL(L.rhoGN(L.id_SL) <= 0 | L.rhoGN(L.id_SL) >= 1);
if ~isempty(bad)
    vn = fwrd_names{bad(1)};
    error('learning:social:badDiscount', ...
        ['"%s" is learned but SL_fitness_discount_%s = %.4g lies outside ' ...
         '(0,1). rhoGN = 0 makes its fitness identically zero: the ' ...
         'tournament degenerates into an unconditional copy and beliefs ' ...
         'random-walk with no selection.'], vn, vn, L.rhoGN(bad(1)));
end

if any(L.mut_sd(L.id_SL) < 0)
    bad = L.id_SL(find(L.mut_sd(L.id_SL) < 0, 1));
    error('learning:social:badMutStd', ...
        'SL_mutation_std_%s = %.4g is negative.', ...
        fwrd_names{bad}, L.mut_sd(bad));
end

if ~isequal(size(L.mut_p), [nfwrd 1])
    error('learning:social:badMutPSize', ...
        'mut_p must be [%d x 1], got [%d x %d].', nfwrd, size(L.mut_p,1), size(L.mut_p,2));
end
bad = L.id_SL(L.mut_p(L.id_SL) <= 0 | L.mut_p(L.id_SL) > 1);
if ~isempty(bad)
    error('learning:social:badMutP', ...
        ['SL_mutation_prob for "%s" is %.4g: the news probability must lie ' ...
         'in (0,1]. The belief loop draws round(mu*J) mutants and rescales ' ...
         'the news by 1/mu.'], fwrd_names{bad(1)}, L.mut_p(bad(1)));
end


if norm(L.Ox*L.Ox - L.Ox, 'fro') > 1e-12
    error('learning:social:OxNotProjector', ...
        'Ox = I - Or''*Or is not idempotent (residual %.3e); check the Or mapping.', ...
        norm(L.Ox*L.Ox - L.Ox, 'fro'));
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
        '(seed %d) so that social-learning streams can use substreams. ' ...
        'Declare set_dynare_seed(''mrg32k3a'', %d) in the .mod to make this ' ...
        'explicit and silence the notice.\n'], prev_algo, seed, seed);
end

% Print SL summary with learned variables
if ~(isfield(options_,'noprint') && options_.noprint)
    declared = @(nm) any(strcmp(M_.param_names, nm));
    src      = {'(default)','(user)   '};

    skipline()
    disp('SOCIAL LEARNING SUMMARY')
    skipline()

    fprintf('  Forward-looking variables:   %d\n', nfwrd);

    if isfield(options_,'DynareRandomStreams')
        fprintf('  RNG:                         %s, seed %d\n', ...
            options_.DynareRandomStreams.algo, options_.DynareRandomStreams.seed);
    end

    skipline()
    fprintf('  %-28s %-10s %-9s %s\n', 'Global option', 'value', 'source', 'parameter');
    fprintf('    %-26s %-10g %s %s\n', 'Agents J',            L.N_agents, src{declared('SL_n_agents')+1},           'SL_n_agents');
    fprintf('    %-26s %-10g %s %s\n', 'Wedding proportion',  L.wed_p,    src{declared('SL_wedding_proportion')+1}, 'SL_wedding_proportion');
    fprintf('    %-26s %-10g %s %s\n', 'Expectation horizon', L.t_max,    src{declared('SL_t_max')+1},              'SL_t_max');
    skipline()

    fprintf('  %-22s %-10s %-12s %-10s %s\n', ...
        'Learned variable','xi (std)','delta (disc)','mu (prob)','news shock');
    for k = 1:L.n_SL
        v  = L.id_SL(k);  vn = fwrd_names{v};
        sh = find(L.Or(v,:) > 0);
        shn = ''; if ~isempty(sh); shn = M_.exo_names{sh}; end
        if declared([prefix_prb vn]); mtag = 'per-var'; else; mtag = 'global'; end
        fprintf('    %-20s %-10.4g %-12.4g %-10.4g %s (mu:%s)\n', ...
            vn, L.mut_sd(v), L.rhoGN(v), L.mut_p(v), shn, mtag);
    end
    re_fwrd = setdiff(1:nfwrd, L.id_SL(:)');
    if ~isempty(re_fwrd)
        fprintf('  %d forward var(s) kept at RE: %s\n', ...
            numel(re_fwrd), strjoin(fwrd_names(re_fwrd), ', '));
    end

    n_news = nnz(any(L.Or,1));
    skipline()
    fprintf('  Shocks: %d structural (Ox) + %d news a_<var> (Or) = %d total\n', ...
        M_.exo_nbr - n_news, n_news, M_.exo_nbr);
    skipline()
end

% return options in dynare global options_.learning.social
L.SL_active = 1;
options_.learning.social = L;
end


%helpers
function v = getp(M_, name, default)
    idx = find(strcmp(M_.param_names, name), 1);
    if isempty(idx) || ~isfinite(M_.params(idx)); v = default; else; v = M_.params(idx); end
end