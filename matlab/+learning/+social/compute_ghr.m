function [oo_, dbg] = compute_ghr(M_, options_, oo_) %#ok<INUSD>
% COMPUTE_GHR  Compute the SL expectations response matrix R (stored as ghr).
%
% INPUTS
%   oo_, M_, options_   standard Dynare structures.
%
% OUTPUTS
%   oo_   updated with oo_.dr.ghr (nvar x nvar, declaration order; only
%         forward-variable columns are non-zero).
%   dbg   [optional] struct with intermediate matrices for validation:
%         .FF .GG .HH .MM .PP .QQ .A .RR  and diagnostics
%         .err_P .err_Q .rcond_A .FR_inf
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

nvar = M_.endo_nbr;

PP = zeros(nvar, nvar);
for ix = 1:nvar
    PP(ix, oo_.dr.state_var) = oo_.dr.ghx(oo_.dr.inv_order_var(ix), :);
end
QQ = oo_.dr.ghu(oo_.dr.inv_order_var, :);


n  = M_.endo_nbr;
ne = M_.exo_nbr;

myss = repmat(oo_.dr.ys, 3, 1);          % 3n x 1
x_ss = zeros(1, ne);

[Jac, ~, ~] = feval([M_.fname '.dynamic_g1'], myss, x_ss, M_.params, oo_.dr.ys, ...
    M_.dynamic_g1_sparse_rowval, M_.dynamic_g1_sparse_colval, ...
    M_.dynamic_g1_sparse_colptr);
if issparse(Jac); Jac = full(Jac); end

HH = Jac(:, 1       : n);          %  y(t-1)
GG = Jac(:, n + 1   : 2*n);        % y(t)
FF = Jac(:, 2*n + 1 : 3*n);        % y(t+1)
MM = Jac(:, 3*n + 1 : 3*n + ne);   % shocks  


A = FF * PP + GG ;
if ~all(isfinite(A(:)))
    error('learning:social:nonfiniteJacobian','Non finite Jacobian in the term FF*PP+GG.');
end
inv_A = A \ eye(nvar);


% calculating R
RR = -(eye(nvar) + inv_A * FF) \ (inv_A * FF);
oo_.dr.ghr = RR;

% Consistency check. 
err_P = max(abs(-inv_A * HH - PP), [], 'all');
err_Q = max(abs(-inv_A * MM - QQ), [], 'all');
scale = max([1, norm(PP, 'inf'), norm(QQ, 'inf')]);
if max(err_P, err_Q) > 1e-8 * scale
    error('learning:social:inconsistentSolution', ...
        ['The SL expectations matrix R could not be validated: the dynamic ' ...
        'Jacobian and the first-order solution are mutually inconsistent ' ...
        '(relative residuals err_P = %.3e, err_Q = %.3e). Check that oo_.dr ' ...
        'was produced by resol for the current M_.params, at order 1 and ' ...
        'without the loglinear option.'], err_P/scale, err_Q/scale);
end

% optional debug output
if nargout > 1
    dbg = struct();
    dbg.FF = FF;  dbg.GG = GG;  dbg.HH = HH;  dbg.MM = MM;
    dbg.PP = PP;  dbg.QQ = QQ;  dbg.A  = A;   dbg.RR = RR;
    dbg.err_P   = err_P;
    dbg.err_Q   = err_Q;
    dbg.rcond_A = rcond(A);
    dbg.FR_inf  = max(abs(FF * RR), [], 'all');
end

end