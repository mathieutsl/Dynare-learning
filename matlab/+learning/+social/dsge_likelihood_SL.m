function [fval,info,exit_flag,DLIK,Hess,model_solution] = dsge_likelihood_SL( ...
        xparam1, dataset_, dataset_info, options_, M_, estim_params_, Prior, ...
        BoundsInfo, dr, endo_steady_state, exo_steady_state, exo_det_steady_state, ...
        smoother_info, derivatives_info, measurement_info)                            %#ok<INUSD>
%DSGE_LIKELIHOOD_SL  Negative log-posterior of a DSGE model under Social Learning.
%%
% INPUTS
% - xparam1             [double]        current values for the estimated parameters.
% - dataset_            [structure]     dataset after transformations
% - dataset_info        [structure]     storing information about the sample
% - options_            [structure]     MATLAB's structure describing the current options
% - M_                  [structure]     MATLAB's structure describing the model
% - estim_params_       [structure]     characterizing parameters to be estimated
% - Prior               [@dprior]       prior information object
% - BoundsInfo          [structure]     containing prior bounds
% - dr                  [structure]     Reduced form model.
% - endo_steady_state   [vector]        steady state value for endogenous variables
% - exo_steady_state    [vector]        steady state value for exogenous variables
% - exo_det_steady_state [vector]       steady state value for exogenous deterministic variables
% - derivatives_info    [structure]     derivative info for identification
%
% OUTPUTS
% - fval                    [double]        scalar, value of minus the likelihood or posterior kernel.
% - info                    [integer]       4×1 vector, information on whether solution and likelihood could be computed
% - exit_flag               [integer]       scalar, equal to 1 (no issues when evaluating the likelihood) or 0 (not able to evaluate the likelihood).
% - DLIK                    [double]        Vector with score of the likelihood
% - Hess                    [double]        asymptotic Hessian matrix.
% - model_solution          [struct]        Structure containing the model solution at the current parameter
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

    % output initialization
    % output initialization
    fval           = [];  %#ok<NASGU>
    info           = zeros(4,1); %#ok<PREALL>
    exit_flag      = 1; %#ok<NASGU>
    DLIK           = [];
    Hess           = [];
    model_solution = [];

    if options_.analytic_derivation
        error('learning:social:analyticDerivation', ...
            'analytic_derivation is not supported under Social Learning.');
    end

    if ~isempty(xparam1)
        xparam1 = xparam1(:);
    end

    %set structural parameters
    M_ = set_all_parameters(xparam1, estim_params_, M_);

    [fval, info, exit_flag, ~, H] = ...
        check_bounds_and_definiteness_estimation(xparam1, M_, estim_params_, BoundsInfo); 
    if info(1)
        return
    end
    if ~isempty(H) && any(H(:) ~= 0)
        error('learning:social:measurementError', ...
            ['Measurement errors are not supported by the SL inversion filter ' ...
            '(#observables must equal #structural shocks).']);
    end

    % refresh estimated SL parameters 
    options_.learning.social = learning.social.refresh_learning_params(options_.learning.social, M_);
    L=options_.learning.social;
    %checking critical SL parameters value range
    if any(L.mut_p(L.id_SL) <= 0) || any(L.mut_p(L.id_SL) > 1) || ...
            any(L.mut_sd(L.id_SL) < 0) || ...
            any(L.rhoGN(L.id_SL) <= 0) || any(L.rhoGN(L.id_SL) >= 1)
        fval = Inf; info(1) = 41; info(4) = 0.1; exit_flag = 0;
        return
    end

    %building statespace
    [dr, info, M_.params] = resol(0, M_, options_, dr, ...
                                  endo_steady_state, exo_steady_state, exo_det_steady_state);
    if info(1)
        fval = Inf; exit_flag = 0;
        return
    end

    oo_.dr = dr;
    oo_    = learning.social.compute_ghr(M_, options_, oo_);

    %Inversion filter
    obs = dataset_.data;                       % T x N

    % RNG
    base_seed  = options_.DynareRandomStreams.seed;
    algo       = options_.DynareRandomStreams.algo;
    belief_rng = learning.learning_streams(base_seed, algo, 'estim');

    [llks, ee, res, y_, stop_dist,expectation] = learning.social.IF_SL(obs, oo_, M_, options_, belief_rng);

    %log likelihood 
    if isfield(options_,'presample') && ~isempty(options_.presample)
        presample = options_.presample;
    else
        presample = 0;
    end
    T  = size(obs,1) - presample;
    T1 = 1 + presample;

    % Restrict to shocks with positive variance: a zero-variance shock makes
    % Sigma_e singular but contributes nothing to the likelihood.
    id_e = find(diag(M_.Sigma_e) > 0);
    ne   = numel(id_e);
    [Rchol, p] = chol(M_.Sigma_e(id_e, id_e));
    if p > 0
        fval = Inf; info(1) = 1; info(4) = 0.1; exit_flag = 0;
        return
    end
    logdetSig = 2*sum(log(diag(Rchol)));
    loglik    = -ne*T/2*log(2*pi) - T/2*logdetSig + sum(llks(T1:end));

    % penalties for inversion breakdown
    if sum(res) < 1e-4
        res = 0;
    end
    if isfield(options_,'penalized_function') && ~isempty(options_.penalized_function)
        pf = options_.penalized_function;
    else
        pf = 1e8;
    end
    loglik = loglik - ( res*pf + stop_dist*pf*10 );

    if isnan(loglik) || isinf(loglik) || ~isreal(loglik)
        fval = Inf; info(1) = 45; info(4) = 0.1; exit_flag = 0;
        return
    end

   
    lnprior = Prior.density(xparam1);
    if isinf(lnprior)
        fval = Inf; info(1) = 40; info(4) = 0.1; exit_flag = 0; return
    end
    if isnan(lnprior)
        fval = Inf; info(1) = 47; info(4) = 0.1; exit_flag = 0; return
    end
    if ~isreal(lnprior)
        fval = Inf; info(1) = 48; info(4) = 0.1; exit_flag = 0; return
    end

    likelihood = -loglik;              
    fval       = likelihood - lnprior; 

    %output for smoother solution
    if nargout >= 6
        model_solution.dr        = oo_.dr;
        model_solution.ys        = oo_.dr.ys;
        model_solution.params    = M_.params;
        model_solution.Sigma_e   = M_.Sigma_e;
        model_solution.learning.social.ee        = ee;
        model_solution.learning.social.y_        = y_;
        model_solution.learning.social.stop_dist = stop_dist;
        model_solution.learning.social.expectation = expectation;
     disp(model_solution)
    end
end
