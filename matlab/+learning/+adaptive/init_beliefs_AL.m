function [beta, Rmom] = init_beliefs_AL(M_, options_,oo_)
% INIT_BELIEFS_AL  Model-consistent initial beliefs (perceived law of motion).
%
% REFERENCES
%   Slobodyan, S. and R. Wouters (2012), "Learning in a Medium-Scale DSGE Model
%   with Expectations Based on Small Forecasting Models", American Economic
%   Journal: Macroeconomics, 4(2), 65-101.
%
%   The recursive least squares updating, the projection facility and the
%   structure of the belief/ALM split follow the algebra of the replication
%   code accompanying that paper. No code was copied verbatim.
%
%   ar1:       beta_k = [c_k, b_k]; intercept and slope are learned
%   intercept: beta_k = [c_k]; constant only (intercept), no slope learned.
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

A  = options_.learning.adaptive;
nb = A.nb; n = M_.endo_nbr;
Q = M_.Sigma_e;

%recovers first order RE policy rule
PP = zeros(n,n);
for ix = 1:n
    PP(ix, oo_.dr.state_var) = oo_.dr.ghx(oo_.dr.inv_order_var(ix), :);
end
QQ = oo_.dr.ghu(oo_.dr.inv_order_var, :);


S  = QQ*Q*QQ';
G0 = reshape((eye(n^2) - kron(PP,PP)) \ S(:), n, n);
G0 = 0.5*(G0+G0');
G1 = PP*G0;                                   % first autocovariance
beta = zeros(A.nfwrd, nb);
Rmom = zeros(nb, nb, A.nfwrd);
for k = 1:A.nfwrd
    j  = A.id_fwrd(k); v0 = G0(j,j);
    if v0 <= 1e-12
        error('learning:adaptive:degenerateVariance', ...
            'Unconditional variance of "%s" is %.3e: cannot initialise the AR(1) belief.', ...
            M_.endo_names{j}, v0);
    end
    if nb == 2                                % ar1: constant + slope
        beta(k,:)   = [0, G1(j,j)/v0];
        Rmom(:,:,k) = [1 0; 0 v0];          
    else                                      % intercept: constant only
        beta(k,:)   = 0;
        Rmom(:,:,k) = 1;                      % E[z z'], z = 1
    end
end
end