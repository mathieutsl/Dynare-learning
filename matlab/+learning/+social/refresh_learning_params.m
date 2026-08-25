function L = refresh_learning_params(L, M_)
%REFRESH_LEARNING_PARAMS  Inject estimated SL parameters into option fields.
%
%   L = refresh_learning_params(L, M_) copies the current values of the
%   estimated Social Learning parameters from M_.params into the option fields
%   consumed by IF_SL. It runs once per likelihood evaluation, on the HOT path,
%   using the index maps precomputed by setup_learning_estimation 
%
% INPUTS
%   L    [struct]  options_.learning, already augmented by setup_learning_estimation.
%   M_   [struct]  model structure with the current M_.params 
%
% OUTPUT
%   L    [struct]  with refreshed mut_sd, rhoGN, mut_p, id_SL.
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

    
    if ~isempty(L.mut_vid)
        L.mut_sd(L.mut_vid) = M_.params(L.mut_pid);
    end
    if ~isempty(L.rhoGN_vid)
        L.rhoGN(L.rhoGN_vid) = M_.params(L.rhoGN_pid);
    end

    % mu:
    if ~isempty(L.mutp_vid)
        L.mut_p(L.mutp_vid) = M_.params(L.mutp_pid);
    end

end
