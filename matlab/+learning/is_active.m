function tf = is_active(options_, mechanism)
% IS_ACTIVE  True if a learning mechanism is configured in options_.
%
% INPUTS
% - options_   [structure]  Dynare options
% - mechanism  [char|cell]  optional, e.g. 'social' or 'adaptive'. When
%                           omitted, true if any mechanism is configured.
% OUTPUTS
% - tf         [logical]    scalar
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

tf = isfield(options_, 'learning') && isstruct(options_.learning);
if ~tf
    return
end
if nargin < 2 || isempty(mechanism)
    mechanism = {'social', 'adaptive'};
elseif ~iscell(mechanism)
    mechanism = {mechanism};
end
tf = false;
for i = 1:numel(mechanism)
    if isfield(options_.learning, mechanism{i}) && ~isempty(options_.learning.(mechanism{i}))
        tf = true;
        return
    end
end
end