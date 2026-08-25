function options_ = setup_learning_estimation(M_, options_, estim_params_) 
%SETUP_LEARNING_ESTIMATION  Build the theta-invariant SL estimation objects.
%
%   options_ = setup_learning_estimation(M_, options_, estim_params_) augments
%   options_.learning .social
%
%   It is the SL equivalent of dynare_estimation_init:run only once
%
% INPUTS
%   M_             [struct]  
%   options_       [struct]  
%   estim_params_  [struct]  
%
% OUTPUT
%   options_.learning augmented with:
%       O, id_obs, id_fwrd, nfwrd, omega,
%       mut_pid, mut_vid, rhoGN_pid, rhoGN_vid,
%       mutp_pid, mutp_vid,
%       estimation (=true).
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

    options_noprint_old = options_.noprint;
    options_.noprint = 1 ;
    skip_simul_checks = true;   % check_support runs in 'estim' mode below
    options_ = learning.social.setup_options_SL(M_, options_, skip_simul_checks);
    options_.noprint = options_noprint_old;
    

    learning.social.check_support(M_, options_, estim_params_, 'estim');
    
    L = options_.learning.social;

    % forward variable
    id_fwrd = find(M_.lead_lag_incidence(3, :) > 0);
    nfwrd   = numel(id_fwrd);
    omega   = zeros(nfwrd, M_.endo_nbr);
    for i = 1:nfwrd
        omega(i, id_fwrd(i)) = 1;
    end
    L.id_fwrd = id_fwrd;
    L.nfwrd   = nfwrd;
    L.omega   = omega;

    % observable selection (declaration order) 
    if ~isfield(options_,'varobs') || isempty(options_.varobs)
        error('learning:social:emptyVarobs', ...
            'options_.varobs is empty; declare observables.');
    end
    vobs   = options_.varobs;
    N      = numel(vobs);
    id_obs = zeros(N, 1);
    for k = 1:N
        idx = find(strcmp(M_.endo_names, vobs{k}));
        if isempty(idx)
            error('learning:social:unknownVarobs', ...
                'varobs "%s" is not an endogenous variable.', vobs{k});
        end
        id_obs(k) = idx;
    end
    O = zeros(N, M_.endo_nbr);
    for k = 1:N
        O(k, id_obs(k)) = 1;
    end
    L.O      = O;
    L.id_obs = id_obs;

    % check inversion constraint : #observables == #shocks 
    if N ~= M_.exo_nbr
        error('learning:social:obsShockMismatch', ...
            ['the inversion filter requires #observables = #shocks ' ...
            '(got %d observables, %d shocks). Balance them by ' ...
            'adding/removing a varobs or the aggregate news shock ' ...
            'a_<var>.'], N, M_.exo_nbr);
    end

    % Parameter index maps, restricted to the learned set.
    pnames     = M_.param_names;
    prefix_mut = 'SL_mutation_std_';
    prefix_dis = 'SL_fitness_discount_';
    prefix_prb = 'SL_mutation_prob_';

    mut_pid   = []; mut_vid   = [];
    rhoGN_pid = []; rhoGN_vid = [];
    mutp_pid  = []; mutp_vid  = [];
    fwrd_names = M_.endo_names(id_fwrd);

    est_pid = [];
    if ~isempty(estim_params_) && isfield(estim_params_, 'param_vals') ...
            && ~isempty(estim_params_.param_vals)
        est_pid = estim_params_.param_vals(:, 1);
    end

    for ip = 1:numel(pnames)
        pn = pnames{ip};
        if strncmp(pn, prefix_mut, numel(prefix_mut))
            kind = 1; vname = pn(numel(prefix_mut)+1:end);
        elseif strncmp(pn, prefix_dis, numel(prefix_dis))
            kind = 2; vname = pn(numel(prefix_dis)+1:end);
        elseif strncmp(pn, prefix_prb, numel(prefix_prb))
            kind = 3; vname = pn(numel(prefix_prb)+1:end);
        else
            continue
        end

        iv = find(strcmp(fwrd_names, vname), 1);
        if isempty(iv)
            continue
        end
        if ~ismember(iv, L.id_SL)
            if ismember(ip, est_pid)
                error('learning:social:orphanEstimParam', ...
                    ['%s is listed in estimated_params but "%s" is not ' ...
                    'learned (SL_learn_%s is absent or 0). The parameter ' ...
                    'does not enter the likelihood.'], pn, vname, vname);
            end
            continue
        end

        switch kind
            case 1
                mut_pid(end+1,1)   = ip; %#ok<AGROW>
                mut_vid(end+1,1)   = iv; %#ok<AGROW>
            case 2
                rhoGN_pid(end+1,1) = ip; %#ok<AGROW>
                rhoGN_vid(end+1,1) = iv; %#ok<AGROW>
            case 3
                mutp_pid(end+1,1)  = ip; %#ok<AGROW>
                mutp_vid(end+1,1)  = iv; %#ok<AGROW>
        end
    end

    L.mut_pid      = mut_pid;
    L.mut_vid      = mut_vid;
    L.rhoGN_pid    = rhoGN_pid;
    L.rhoGN_vid    = rhoGN_vid;
    L.mutp_pid     = mutp_pid;
    L.mutp_vid     = mutp_vid;

    if ~isequal(size(L.mut_p), [nfwrd 1])
        error('learning:social:badMutP', ...
              ['options_.learning.social.mut_p must be [%d x 1], got [%d x %d]. ' ...
               'Run setup_options_SL first.'], nfwrd, size(L.mut_p,1), size(L.mut_p,2));
    end

    
    % Recap SL setup
    if ~(isfield(options_,'noprint') && options_.noprint)
        endo     = M_.endo_names;
        fwrd_nm  = endo(L.id_fwrd);
        N_obs    = numel(L.id_obs);

        skipline()
        disp('SOCIAL LEARNING ESTIMATION SUMMARY')
        skipline()

        fprintf('  Forward-looking variables:   %d\n', L.nfwrd);
        fprintf('  Learned variables:           %d / %d  (%s)\n', ...
            L.n_SL, L.nfwrd, strjoin(fwrd_nm(L.id_SL(:)'), ', '));
        if isfield(options_,'DynareRandomStreams')
            fprintf('  RNG:                         %s, seed %d\n', ...
                options_.DynareRandomStreams.algo, options_.DynareRandomStreams.seed);
        end

        % Observability structure
        skipline()
        fprintf('  Observables:                 %d obs  vs  %d exo shocks', ...
            N_obs, M_.exo_nbr);
        if N_obs == M_.exo_nbr
            fprintf('  --> inversion OK\n');
        else
            fprintf('  --> MISMATCH (IF_SL requires N_obs == exo_nbr)\n');
        end

        skipline()
        fprintf('  Observable variables:\n');
        for k = 1:N_obs
            fprintf('    %s\n', endo{L.id_obs(k)});
        end

        % Estimated vs. calibrated.
        n_mut    = numel(intersect(L.mut_pid,   est_pid));
        n_rhoGN  = numel(intersect(L.rhoGN_pid, est_pid));
        n_mutp_v = numel(intersect(L.mutp_pid,  est_pid));

        skipline()
        fprintf('  SL parameters in estimated_params (out of %d learned variable(s)):\n', L.n_SL);
        fprintf('    xi     (SL_mutation_std_<v>)     : %d\n', n_mut);
        fprintf('    delta  (SL_fitness_discount_<v>) : %d\n', n_rhoGN);
        fprintf('    mu     (SL_mutation_prob_<v>)    : %d\n', n_mutp_v);

        % Per-variable parameter mapping
        skipline()
        fprintf('  %-16s %-28s %-28s %s\n', ...
            'Learned var','xi param','delta param','mu param');
   
        for k = 1:L.n_SL
            iv = L.id_SL(k);
            vn = fwrd_nm{iv};
            fprintf('    %-14s %-28s %-28s %s\n', vn, ...
                tag(L.mut_pid(L.mut_vid == iv),     M_, est_pid), ...
                tag(L.rhoGN_pid(L.rhoGN_vid == iv), M_, est_pid), ...
                tag(L.mutp_pid(L.mutp_vid == iv),   M_, est_pid));
        end
        skipline()
    end

    L.estimation      = true;
    options_.learning.social = L;
end

% Helper
% Helper
function s = tag(pid, M_, est_pid)
s = M_.param_names{pid};
if ~ismember(pid, est_pid)
    s = [s '  (calibrated)'];
end
end