function plot_decomp(oo_, M_, varlist, options_)
% PLOT_DECOMP  Plot the historical shock decomposition under Social Learning.
%   learning.social.plot_decomp(oo_, M_, {'y','pi','i','pie'}, options_)
%
% INPUTS
%   oo_       [struct]     must contain oo_.learning.social.decomp [nvar x T x n_chan]
%                          and oo_.learning.social.decomp_names {1 x n_chan}
%   M_        [struct]     Dynare model
%   varlist   [cell]       variable names to plot (e.g. {'y','pi','i'})
%   options_  [struct]     optional; uses options_.learning.social.decomp_labels
%                         
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


res    = learning.getopt(learning.getopt(oo_, 'learning', struct()), 'social', struct());
decomp = learning.getopt(res, 'decomp', []);
if isempty(decomp)
    error('learning:social:noDecomp', ...
        ['oo_.learning.social.decomp not found. Run stoch_simul with ' ...
        'options_.learning.social.decomp = true.']);
end

decomp      = oo_.learning.social.decomp;          % [nvar x T+1 x n_chan]
chan_names   = oo_.learning.social.decomp_names;    % {1 x n_chan}
n_chan_ee    = numel(oo_.learning.social.decomp_id_ee);
n_chan       = size(decomp, 3);
Tplot        = size(decomp, 2);

% reorder: beliefs first, then shocks 
idx_belief = (n_chan_ee + 1):n_chan;
idx_shock  = 1:n_chan_ee;
plot_order  = [idx_belief, idx_shock];
decomp      = decomp(:, :, plot_order);
chan_names   = chan_names(plot_order);

% Optional settings
L = struct();
if nargin >= 4 && learning.is_active(options_, 'social')
    L = options_.learning.social;
end

% percentage scaling (default: on)
if learning.getopt(L, 'decomp_pct', true)
    scale    = 100;
    unit_str = ' (%)';
else
    scale    = 1;
    unit_str = '';
end

% legend entries: user-supplied, or auto-generated from the channel names
leg_labels = learning.getopt(L, 'decomp_labels', {});
if numel(leg_labels) ~= n_chan
    if ~isempty(leg_labels)
        warning('learning:social:decompLabels', ...
            ['options_.learning.social.decomp_labels has %d entries for %d ' ...
            'channels; falling back to automatic labels. Expected order: ' ...
            'belief channels first, then structural shocks.'], ...
            numel(leg_labels), n_chan);
    end
    prefix     = 'belief_';
    leg_labels = chan_names;
    for k = 1:n_chan
        if startsWith(chan_names{k}, prefix)
            leg_labels{k} = ['SL expectations (' chan_names{k}(numel(prefix)+1:end) ')'];
        end
    end
end

colors = get(groot, 'defaultAxesColorOrder');
if n_chan > size(colors, 1)
    colors = [colors; lines(n_chan - size(colors, 1))];
end

% time axis
tvec = 1:Tplot;

% plotting
for iv = 1:numel(varlist)
    idx_var = find(strcmp(M_.endo_names, varlist{iv}));
    if isempty(idx_var)
        warning('learning:social:unknownVariable', 'variable "%s" not found, skipping.', varlist{iv})
        continue;
    end

    % contributions matrix: [T x n_chan]
    C = scale * squeeze(decomp(idx_var, :, :));

    figure('Name', ['Historical decomposition: ' varlist{iv}]);
    hold on;

    h_patches = gobjects(n_chan, 1);

    % positive contributions
    cum_pos = zeros(Tplot, 1);
    cum_neg = zeros(Tplot, 1);
    hw = 0.4;   

    for k = 1:n_chan
        y_bot = zeros(Tplot, 1);
        y_top = zeros(Tplot, 1);

        for t = 1:Tplot
            if C(t, k) >= 0
                y_bot(t) = cum_pos(t);
                y_top(t) = cum_pos(t) + C(t, k);
                cum_pos(t) = y_top(t);
            else
                y_top(t) = cum_neg(t);
                y_bot(t) = cum_neg(t) + C(t, k);
                cum_neg(t) = y_bot(t);
            end
        end

        for t = 1:Tplot
            if abs(C(t, k)) > 1e-12
                fill([t-hw; t-hw; t+hw; t+hw], ...
                     [y_bot(t); y_top(t); y_top(t); y_bot(t)], ...
                     colors(k,:), 'EdgeColor', 'none');
            end
        end

        
        h_patches(k) = patch(NaN, NaN, colors(k,:), 'EdgeColor', 'none');
    end

    % total trajectory overlay
    y_total = scale * (oo_.endo_simul(idx_var, :) - oo_.dr.ys(idx_var));
    plot(tvec, y_total(1:Tplot), 'k-', 'LineWidth', 1.5);

    % zero line
    plot(tvec([1 end]), [0 0], 'k-', 'LineWidth', 0.5);

    hold off;
    xlim([tvec(1) - 1, tvec(end) + 1]);
    legend(h_patches, leg_labels, 'Interpreter', 'latex', 'Location', 'best');
    ylabel(['Deviation from steady state' unit_str]);
    title(['Historical decomposition: ' varlist{iv}], 'Interpreter', 'none');
    grid on;
    box on;
end

end
