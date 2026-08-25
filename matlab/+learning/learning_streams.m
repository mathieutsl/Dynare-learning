function [belief, shock] = learning_streams(base_seed, algo, mode, k)
% One seed, two non-overlapping substreams to create two independent streams.
%   mode='sim'   : substream indexed by k (intended variation across replications)
%   mode='estim' : FIXED substream -> frozen belief realization (CRN)
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

if nargin < 4 || isempty(k); k = 1; end
if ~ismember(algo, {'mrg32k3a','mlfg6331_64'})
    warning('learning:promotedAlgo', ...
        ['RNG "%s" has no substreams; the learning streams are built on ' ...
        'mrg32k3a (seed %d) instead. The global Dynare stream is unchanged. ' ...
        'Declare set_dynare_seed(''mrg32k3a'', %d) in the .mod to silence ' ...
        'this warning.'], algo, base_seed, base_seed);
    algo = 'mrg32k3a';
end
belief = RandStream(algo, 'Seed', base_seed);
shock  = RandStream(algo, 'Seed', base_seed);
switch mode
    case 'sim'
        belief.Substream = 2*k - 1;
        shock.Substream  = 2*k;
    case 'estim'
        belief.Substream = 1;   % Independant of MH draws
        shock.Substream  = 2;  
    otherwise
        error('learning:badMode', 'mode should be either ''sim'' or ''estim''.');
end
end