function oo_ = simult_SL(M_, options_, oo_, ee, belief_rng)
% SIMULT_SL  Simulate the model under Social Learning expectations.
%
% Policy function under SL:
%   y(t) = ys + P*(y(t-1)-ys) + Q*Ox*e(t) + R(:,id_fwrd)*(phi(t) + Or*e(t))
%
% INPUTS
%   ee         [T x nexo]  exogenous shock realizations 
%   oo_        [struct]     Dynare results
%   M_         [struct]
%   options_   [struct]     options_.learning.social 
%   belief_rng [RandStream] RNG that drives the belief path (mutation news +
%                           tournament pairings). 
% OUTPUTS
%   oo_.endo_simul                  [nvar x T]
%   oo_.learning.social.beliefs            [nfwrd x J x T]   post-tournament a_{j,t}
%   oo_.learning.social.beliefs_candidate  [nfwrd x J x T]   post-injection m_{j,t}
%   oo_.learning.social.fitness            [nfwrd x J x T]
%   oo_.learning.social.expectations       [nfwrd x T]       aggregate phi_t + Or*e(t)
%   oo_.learning.social.expectations_ahead [nfwrd x T x t_max]
%   oo_.learning.social.final_beliefs      [nfwrd x J]
%   oo_.learning.social.y_RE               [nvar x  T] rational expectation trajectory
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


if nargin < 5 || isempty(belief_rng)
    error('learning:social:noStream', ...
        ['belief_rng (5th arg) is required. Build it with ' ...
         'learning.learning_streams(options_.DynareRandomStreams.seed, ' ...
         'options_.DynareRandomStreams.algo, ''sim'', k). It is deliberately ' ...
         'not derived from options_ so that parfor substreams and estimation ' ...
         'CRN stay under caller control.']);
end
if ~isa(belief_rng, 'RandStream')
    error('learning:social:badStream', 'belief_rng must be a RandStream, got %s.', class(belief_rng));
end

T       = size(ee, 1);
nvar    = M_.endo_nbr;
nexo    = M_.exo_nbr;
y0      = oo_.dr.ys;

J       = options_.learning.social.N_agents;
mut_p   = options_.learning.social.mut_p;      % [nfwrd x 1], mu
wed_p   = options_.learning.social.wed_p;      % scalar
mut_sd  = options_.learning.social.mut_sd;     % [nfwrd x 1], xi
rhoGN   = options_.learning.social.rhoGN;      % [nfwrd x 1], delta
t_max   = options_.learning.social.t_max;
Or      = options_.learning.social.Or;
Ox      = options_.learning.social.Ox;

id_fwrd = find(M_.lead_lag_incidence(3, :) > 0);
nfwrd   = length(id_fwrd);

if ~isequal(size(mut_p), [nfwrd 1])
    error('learning:social:badMutPSize', ...
        ['options_.learning.social.mut_p must be [%d x 1], got [%d x %d]. ' ...
        'Run setup_options_SL.'], ...
          nfwrd, size(mut_p, 1), size(mut_p, 2));
end

nb_mut = min(J, round(mut_p .* J));    
nb_wed = 2*floor(wed_p * J / 2);       

UU = ones(J, 1) / J;

omega = zeros(nfwrd, nvar);
for k = 1:nfwrd
    omega(k, id_fwrd(k)) = 1;
end

% Policy-function matrices
PP = zeros(nvar, nvar);
for ix = 1:nvar
    PP(ix, oo_.dr.state_var) = oo_.dr.ghx(oo_.dr.inv_order_var(ix), :);
end
QQ = oo_.dr.ghu(oo_.dr.inv_order_var, :);
RR = oo_.dr.ghr;

% Fitness blocks
Pf   = omega * PP;                  % [nfwrd x nvar]
R_ff = omega * RR(:, id_fwrd);      % [nfwrd x nfwrd]
rho_mat = rhoGN .^ (1:500);
rho_mat = fliplr(rho_mat(:, 1:max(sum(rho_mat > 0.01, 2))))';
T0      = size(rho_mat, 1);
W       = rho_mat(end:-1:1, :).';   % [nfwrd x T0]

% Initial beliefs
if isfield(options_.learning.social, 'initial_beliefs') && ~isempty(options_.learning.social.initial_beliefs)
    a_j = options_.learning.social.initial_beliefs;          % [nfwrd x J]
    if size(a_j, 1) ~= nfwrd || size(a_j, 2) ~= J
        error('learning:social:badInitialBeliefs', ...
            'initial_beliefs must be [%d x %d], got [%d x %d]', ...
              nfwrd, J, size(a_j, 1), size(a_j, 2));
    end
else
    a_j = zeros(nfwrd, J);     
end

% Storage
Ttotal   = T0 + T + M_.maximum_lag;
y_ = repmat(y0, 1, Ttotal);   
Fitness  = zeros(nfwrd, J, Ttotal);
i0 = T0 + M_.maximum_lag + 1;
y_RE = zeros(size(y_));
y_RE(:, i0-1) = y_(:, i0-1); 

beliefs_store       = zeros(nfwrd, J, T);   % post-tournament a_j
candidate_store     = zeros(nfwrd, J, T);   % post-injection m_j 
fitness_store       = zeros(nfwrd, J, T);
expectations_store  = zeros(nfwrd, T);
expect_ahead_store  = zeros(nfwrd, T, t_max);

id_mut = cell(nfwrd, 1);

% shock_decomposition (optional)
do_decomp = logical(learning.getopt(options_.learning.social, 'decomp', false));
if do_decomp
    id_ee_real = find(diag(Ox) > 0.5);         
    n_chan_ee   = numel(id_ee_real);
    n_chan_aa   = nfwrd;                        
    n_chan      = n_chan_ee + n_chan_aa;
    decomp     = zeros(nvar, Ttotal, n_chan);    % [nvar x Ttotal x n_chan]
end


for i = (T0 + M_.maximum_lag + 1):Ttotal

    t_ee = i - M_.maximum_lag - T0;          

    for v = 1:nfwrd
        perm_v    = randperm(belief_rng, J);
        id_mut{v} = perm_v(1:nb_mut(v));
    end
    jump_sd = repmat(mut_sd, [1, J]) .* randn(belief_rng, nfwrd, J);

    m_j = a_j;
    for v = 1:nfwrd
        m_j(v, id_mut{v}) = m_j(v, id_mut{v}) + jump_sd(v, id_mut{v});
    end

    % candidate aggregate that enters the policy function
    phi_t = m_j * UU;
    y_(:, i) = y0 + PP * (y_(:, i-1) - y0) ...
             + QQ * Ox * ee(t_ee, :)' ...
             + RR(:, id_fwrd) * (phi_t + Or * ee(t_ee, :)');
   

    Ore = Or * ee(t_ee, :)';                % [nfwrd x 1]
    for v = 1:nfwrd
        m_j(v, id_mut{v}) = m_j(v, id_mut{v}) + Ore(v) / mut_p(v);
    end

    % decomposition
    if do_decomp
        for ix = 1:n_chan_ee
            eex = zeros(nexo, 1);
            eex(id_ee_real(ix)) = ee(t_ee, id_ee_real(ix));
            decomp(:, i, ix) = PP * decomp(:, i-1, ix) + QQ * Ox * eex;
        end
        for ia = 1:n_chan_aa
            phi_ia = m_j(ia, :) * UU;   
            decomp(:, i, n_chan_ee + ia) = PP * decomp(:, i-1, n_chan_ee + ia) ...
                                           + RR(:, id_fwrd(ia)) * phi_ia;
        end
    end

    % Fitness                       
    F  = zeros(nfwrd, J);
    for it_ = 1:T0
        err = m_j + Pf*(y_(:,i-it_)-y0) + R_ff*m_j - (y_(id_fwrd,i-it_+1) - y0(id_fwrd)) ;
        F   = F - W(:, it_) .* err.^2;
    end
    Fitness(:, :, i) = F;

    % Tournament 
    perm_wed   = randperm(belief_rng, J);
    id_wedding = reshape(perm_wed(1:nb_wed), [nb_wed/2, 2]);
    for varJJ = 1:nfwrd
        best_bool = Fitness(varJJ, id_wedding(:,1), i) > Fitness(varJJ, id_wedding(:,2), i);
        a_j(varJJ, id_wedding(:,1)) = best_bool  .* m_j(varJJ, id_wedding(:,1)) ...
                                    + (~best_bool) .* m_j(varJJ, id_wedding(:,2));
        a_j(varJJ, id_wedding(:,2)) = a_j(varJJ, id_wedding(:,1));
    end

    % storing
    if t_ee >= 1 && t_ee <= T
        beliefs_store(:, :, t_ee)   = a_j;                 % post-tournament
        candidate_store(:, :, t_ee) = m_j;                 % post-injection candidate
        fitness_store(:, :, t_ee)   = F;
        expectations_store(:, t_ee) = phi_t + Or * ee(t_ee, :)';

        % h-step-ahead aggregate SL expectation
        abar  = mean(a_j, 2);                              % [nfwrd x 1], phi_t
        wedge = zeros(nvar, 1);
        wedge(id_fwrd) = abar;
        yhat  = y_(:, i) - y0;                            
        for h = 1:t_max
            if h > 1
                yhat = yhat + wedge;                       % belief carried into the state
            end
            yhat = PP * yhat + RR(:, id_fwrd) * abar;
            expect_ahead_store(:, t_ee, h) = abar + omega * yhat;
        end
    end

end

% Build rational expectation trajectory for same shocks 
for i = (T0+M_.maximum_lag+1):(T0+T+M_.maximum_lag)
    y_RE(:,i) = y0 + PP*(y_RE(:,i-1)-y0) + QQ*Ox*ee(i-M_.maximum_lag-T0,:)';
end


%saving output :
oo_.endo_simul = y_(:, (M_.maximum_lag + T0 +1 ):end,:);    
if do_decomp
    oo_.learning.social.decomp = decomp(:, (M_.maximum_lag + T0 + 1):end, :);
    chan_names = cell(1, n_chan);
    for ix = 1:n_chan_ee
        chan_names{ix} = M_.exo_names{id_ee_real(ix)};
    end
    for ia = 1:n_chan_aa
        chan_names{n_chan_ee + ia} = ['belief_' M_.endo_names{id_fwrd(ia)}];
    end
    oo_.learning.social.decomp_names  = chan_names;
    oo_.learning.social.decomp_id_ee  = id_ee_real;
    oo_.learning.social.decomp_id_aa  = (1:n_chan_aa);
end

oo_.learning.social.beliefs            = beliefs_store;
oo_.learning.social.beliefs_candidate  = candidate_store;
oo_.learning.social.fitness            = fitness_store;
oo_.learning.social.expectations       = expectations_store;
oo_.learning.social.expectations_ahead = expect_ahead_store;
oo_.learning.social.final_beliefs      = a_j;
oo_.learning.social.y_RE = y_RE(:, (M_.maximum_lag + T0 +1 ):end,:);
end