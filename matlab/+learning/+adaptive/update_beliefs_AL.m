function [beta, Rmom] = update_beliefs_AL(beta, Rmom, yf_now, z_lag, g)
% UPDATE_BELIEFS_AL  Constant-gain RLS belief updating .

% REFERENCES
%   Slobodyan, S. and R. Wouters (2012), "Learning in a Medium-Scale DSGE Model
%   with Expectations Based on Small Forecasting Models", American Economic
%   Journal: Macroeconomics, 4(2), 65-101.
%
%   The recursive least squares updating, the projection facility and the
%   structure of the belief/ALM split follow the algebra of the replication
%   code accompanying that paper. No code was copied verbatim.
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

for j = 1:size(beta,1)
    z   = z_lag(:,j);
    Rj  = Rmom(:,:,j) + g*(z*z' - Rmom(:,:,j));
    err = yf_now(j) - z'*beta(j,:)';
    beta(j,:)   = beta(j,:) + (g*(Rj\(z*err)))';
    Rmom(:,:,j) = Rj;
end
end