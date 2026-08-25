function [T,R,mu,ok] = resolve_alm_AL(beta, J, M_, options_)
% RESOLVE_ALM_AL  Map current beliefs into the Actual Law of Motion (ALM).
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
n  = M_.endo_nbr;
B0 = zeros(n); cvec = zeros(n,1);
for k = 1:A.nfwrd
    j = A.id_fwrd(k);
    cvec(j) = beta(k,1);                      % perceived constant (both modes)
    if A.nb == 2                              % ar1: perceived slope
        B0(j,j) = beta(k,2);
    end
end
Amat = J.GG + J.FF*B0;
if ~all(isfinite(Amat(:))) || rcond(Amat) < 1e-12
    T=[]; R=[]; mu=[]; ok=false; return
end
iA = Amat \ eye(n);
T  = -iA*J.HH;   R = -iA*J.MM;   mu = -iA*(J.FF*cvec);
ok = max(abs(eig(T))) < A.pj_threshold;       % projection facility (0-1)
end