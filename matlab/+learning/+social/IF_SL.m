function [llks,ee,res,y_,stop_dist,expectations] = IF_SL(obs, oo, M, options_, belief_rng)
%IF_SL  Inversion filter under Social Learning expectations
%
% INPUTS
%   obs        [T x N]     observed data:  N must equal M.exo_nbr (inversion constraint).
%   oo         [struct]    Dynare results; requires oo.dr.
%   M          [struct]    model structure.
%   options_   [struct]    requires learning field in options
%   belief_rng [RandStream] RNG driving the belief path (mutation news +
%                          tournament pairings). .
%
% OUTPUTS
%   llks       [T x 1]     per-period log-likelihood contributions.
%   ee         [T x N]     extracted structural shocks (smoothed shocks).
%   res        [scalar]    cumulated inversion residual (penalty signal).
%   y_         [endo x .]  smoothed state trajectory (declaration order).
%   stop_dist  [scalar]    periods before T at which inversion was aborted
%   expectations [nfwrd x T]  aggregate SL expectations entering the policy
%                             function: phi_t + Or*e(t). 
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


    L = options_.learning.social;
    
    %RNG
    if nargin < 5 || isempty(belief_rng)
        error('learning:social:noStream', ...
            ['belief_rng (5th arg) is required. dsge_likelihood_SL builds it ' ...
             'with learning.learning_streams(options_.DynareRandomStreams.seed, ' ...
             'options_.DynareRandomStreams.algo, ''estim''). It is not derived ' ...
             'from options_ so that estimation CRN stays under caller control.']);
    end
    if ~isa(belief_rng, 'RandStream')
        error('learning:social:badStream', 'belief_rng must be a RandStream, got %s.', class(belief_rng));
    end

    % selection matrices & dimensions
    O       = L.O;                 % N x endo, declaration order
    Or      = L.Or;                % nfwrd x exo
    Ox      = L.Ox;                % exo x exo
    id_fwrd = L.id_fwrd;
    nfwrd   = L.nfwrd;
    omega   = L.omega;             % nfwrd x endo
    N       = numel(L.id_obs);
    T       = size(obs,1);
    y0      = oo.dr.ys;
    expectations = zeros(nfwrd, T);

    if size(obs,2) ~= N
        error('learning:social:obsMismatch', ...
            'obs has %d columns but %d observables are declared.', size(obs,2), N);
    end
    
    if T < 1
        error('learning:social:emptySample', 'obs is empty.');
    end
    if any(~isfinite(obs(:)))
        error('learning:social:missingData', ...
            ['The inversion filter cannot handle missing observations ' ...
            '(%d non-finite values in the data).'], sum(~isfinite(obs(:))));
    end

    % state space: R from ghr, P and Q by re-indexing 
    RR = oo.dr.ghr;                                              % endo x endo
    PP = zeros(M.endo_nbr);
    PP(:, oo.dr.state_var) = oo.dr.ghx(oo.dr.inv_order_var, :);  % endo x endo
    QQ = oo.dr.ghu(oo.dr.inv_order_var, :);                      % endo x exo
    RRf = RR(:, id_fwrd);                                        % endo x nfwrd

    % fitness blocks invariant objects
    Pf   = omega * PP;                                           % nfwrd x endo
    R_ff = omega * RRf;                                          % nfwrd x nfwrd

    % Social Learning config
    J     = L.N_agents;
    mut_p = L.mut_p(:);                                          % nfwrd x 1, mu
    rhoGN = L.rhoGN(:);                                          % nfwrd x 1, delta

    % discounted-fitness weights
    rho_mat = rhoGN.^(1:500);
    rho_mat = fliplr(rho_mat(:, 1:max(sum(rho_mat > 0.01, 2))))';
    T0 = size(rho_mat, 1);              
    W  = rhoGN(:) .^ (1:T0); 

    UU = ones(J,1) / J;            

    % Jacobian 
    lambda        = O*QQ*Ox + O*RRf*Or;      % N x N, invariant across periods
    id_e  = find(diag(M.Sigma_e) > 0);
    S     = M.Sigma_e(id_e, id_e);
    if N ~= M.exo_nbr
        error('learning:social:obsShockMismatch', ...
            ['The inversion filter requires #observables (%d) == #shocks (%d). ' ...
            'ee columns are indexed in shock space.'], N, M.exo_nbr);
    end
    [Rc, p] = chol(S);
    if p ~= 0
        error('learning:social:singularSigmaE', ...
            ['Sigma_e restricted to shocks with positive variance is not ' ...
            'positive definite (leading minor %d).'], p);
    end           

    rc = rcond(lambda);
    if ~isfinite(rc) || rc < 1e-12
        error('learning:social:illConditionedLambda', ...
            ['The inversion Jacobian lambda = O*Q*Ox + O*R_f*Or is ill-conditioned ' ...
            '(rcond = %.3e): the observables carry almost no independent ' ...
            'information about the shocks.'], rc);
    end

    inv_lambda = lambda \ eye(N);
    [~, Ulam]  = lu(lambda);
    logdet_lambda = sum(log(abs(diag(Ulam))));


    if isfield(options_,'prefilter') && options_.prefilter
        cst = zeros(N, 1);
    else
        cst = O * y0;
    end

   
    a_j     = zeros(nfwrd, J);   %belief deviation initialized to 0

    y_        = repmat(y0, 1, T0 + T + M.maximum_lag);   %starting at steady state
    ee        = zeros(T, N);
    llks      = zeros(T, 1);
    res       = 0;

    n_mut = min(J, round(mut_p .* J));
    n_wed = 2*floor(L.wed_p * J / 2);                   
    id_mut  = cell(nfwrd, 1);
   

    % forward inversion loop
    for i = (T0 + M.maximum_lag + 1):(T0 + T + M.maximum_lag)

        t = i - M.maximum_lag - T0;    

        for v = 1:nfwrd
            perm_v    = randperm(belief_rng, J);
            id_mut{v} = perm_v(1:n_mut(v));
        end
        jump_sd = repmat(L.mut_sd, [1, J]) .* randn(belief_rng, nfwrd, J);

        m_j = a_j;
        for v = 1:nfwrd
            m_j(v, id_mut{v}) = m_j(v, id_mut{v}) + jump_sd(v, id_mut{v});
        end

        phi_t = m_j * UU;                                  % nfwrd x 1
        % INVERSION : extract shocks given current beliefs
        ee(t,:) = ( inv_lambda * (obs(t,:)' - cst - O*( PP*(y_(:,i-1)-y0) + RRf*phi_t )) )';

         % aggregate expectations entering the policy function (belief + news),
        expectations(:, t) = phi_t + Or * ee(t,:)';

        % TRANSITION : new state given shocks and beliefs
        y_(:,i) = y0 + PP*(y_(:,i-1) - y0) + QQ*Ox*ee(t,:)' + RRf*expectations(:,t);

        Ore = Or*ee(t,:)';                   % nfwrd x 1
        for v = 1:nfwrd
            m_j(v, id_mut{v}) = m_j(v, id_mut{v}) + Ore(v) / mut_p(v);
        end

        % FITNESS 
        Rm = R_ff * m_j;                     % nfwrd x J
        F  = zeros(nfwrd, J);
        for it_ = 1:T0
            Py  = Pf * (y_(:,i-it_) - y0);            
            yf  = y_(id_fwrd, i-it_+1) - y0(id_fwrd); 
            err = (m_j + (Py + Rm)) - yf;
            F   = F - W(:,it_) .* err.^2;
        end
    
        % tournament
        perm       = randperm(belief_rng, J);
        sel        = perm(1:n_wed);
        id_wedding = [sel(1:n_wed/2)', sel(n_wed/2+1:n_wed)'];
        for varJJ = 1:nfwrd
            best   = F(varJJ, id_wedding(:,1)) > F(varJJ, id_wedding(:,2));
            winner = best .* m_j(varJJ, id_wedding(:,1)) + (~best) .* m_j(varJJ, id_wedding(:,2));
            a_j(varJJ, id_wedding(:,1)) = winner;
            a_j(varJJ, id_wedding(:,2)) = winner;
        end

        % log likelihood
        z       = Rc' \ ee(t, id_e)';
        llks(t) = -0.5 * (z' * z) - logdet_lambda;

        % Res
        res = res + sum(abs( obs(t,:)' - cst - O*(y_(:,i) - y0) ));
        if res > 1
            break
        end       
    end

    t_last    = i - M.maximum_lag - T0;
    stop_dist = T - t_last;  
    y_        = y_(:, (M.maximum_lag + T0):end);
end
