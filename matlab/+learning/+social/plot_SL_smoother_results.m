function plot_SL_smoother_results(oo_, M_, options_, dataset_)
%PLOT_SL_SMOOTHER_RESULTS  Plot the IF_SL smoother output.
%   plot_SL_smoother_results(oo_, M_, options_, dataset_)
%
%   INPUTS
%     oo_       results, with oo_.SmoothedShocks, oo_.SmoothedVariables
%     M_        model structure
%     options_  Dynare options 
%     dataset_  optional
%
%   OUTPUTS
%   Figures:
%     1. Smoothed structural shocks                -> <fname>_SL_SmoothedShocks_*
%     2. Smoothed observables overlaid on the data -> <fname>_SL_SmoothedObs_*
%     3. Aggregate news / belief channel (Or)      -> <fname>_SL_BeliefNews_*
%     4. Aggregate SL expectations                 -> <fname>_SL_Expectations_*
%
%   Periods the inversion did not reach are NaN and simply appear as gaps.
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

if isfield(options_, 'nograph') && options_.nograph
    return
end
if nargin < 4
    dataset_ = [];
end

DirName   = CheckPath('graphs', M_.dname);
nodisplay = options_.nodisplay;
gfmt      = options_.graph_format;
fname     = M_.fname;

% smoothed structural shocks
sh = M_.exo_names;
Ys = cellfun(@(n) oo_.SmoothedShocks.(n)(:), sh, 'UniformOutput', false);
paged_plot(sh, Ys, [], {}, 'SL smoothed shocks', ...
           [fname '_SL_SmoothedShocks'], DirName, nodisplay, gfmt);

% smoothed observables, overlaid on the data
vo = options_.varobs;
if ~iscell(vo)
    vo = cellstr(vo);
end
Yv   = cellfun(@(n) oo_.SmoothedVariables.(n)(:), vo, 'UniformOutput', false);
T_sm = numel(Yv{1});
Dov = [];
if ~isempty(dataset_) && size(dataset_.data, 1) >= T_sm
    Dov = dataset_.data(end-T_sm+1:end, :);
end
paged_plot(vo, Yv, Dov, {'smoothed', 'data'}, 'SL smoothed vs data', ...
           [fname '_SL_SmoothedObs'], DirName, nodisplay, gfmt);

% aggregate news (belief channel)
Or = [];
if learning.is_active(options_, 'social')
    Or = learning.getopt(options_.learning.social, 'Or', []);
end
news_cols = find(any(Or ~= 0, 1));
if ~isempty(news_cols)
    nn = M_.exo_names(news_cols);
    Yn = cellfun(@(n) oo_.SmoothedShocks.(n)(:), nn, 'UniformOutput', false);
    paged_plot(nn, Yn, [], {}, 'SL aggregate news (belief channel)', ...
               [fname '_SL_BeliefNews'], DirName, nodisplay, gfmt);
end

% aggregate SL expectations against the realized forward variables
sm = [];
if learning.is_active(options_, 'social')
    sm = learning.getopt(oo_.learning.social, 'smoother', []);
end
if ~isempty(sm) && isfield(sm, 'expectations')
    id_SL = learning.getopt(options_.learning.social, 'id_SL', []);
    if isempty(id_SL)
        id_SL = 1:numel(sm.forward_vars);
    end
    nm = cellfun(@(v) ['belief wedge on ' v], sm.forward_vars(id_SL), 'UniformOutput', false);
    Ye = arrayfun(@(k) sm.expectations(:, k), id_SL(:)', 'UniformOutput', false);
    paged_plot(nm, Ye, [], {}, 'SL aggregate belief wedge', ...
        [fname '_SL_BeliefWedge'], DirName, nodisplay, gfmt);
end
end 

%helper function
function paged_plot(names, Y, Overlay, leg, ttl, fbase, DirName, nodisplay, gfmt)

n = numel(names);
if n == 0
    return
end

[nbplt, nr, nc, ~, ~, nstar] = pltorg(n);
T = numel(Y{1});
t = (1:T)';

col_main    = [0 102 204]/255;
col_overlay = [0.85 0 0];
col_zero    = [0.10 0.10 0.10];

idx = 0;
for pg = 1:nbplt
    hh = dyn_figure(nodisplay, 'Name', sprintf('%s (%d)', ttl, pg));
    np_this = min(nstar, n - nstar*(pg-1));
    for k = 1:np_this
        idx = idx + 1;
        subplot(nr, nc, k);
        plot([t(1) t(end)], [0 0], ':', 'Color', col_zero, ...
             'LineWidth', 1.0, 'HandleVisibility', 'off');
        hold on
        plot(t, Y{idx}, '-', 'Color', col_main, 'LineWidth', 1.3);
        if ~isempty(Overlay)
            plot(t, Overlay(:, idx), '--', 'Color', col_overlay, 'LineWidth', 1.0);
            if idx == 1 && numel(leg) == 2
                legend(leg, 'Location', 'best', 'Box', 'off');
            end
        end
        title(names{idx}, 'Interpreter', 'none');
        if any(isfinite(Y{idx}))
            axis tight
        end
        box on
        hold off
    end
    dyn_saveas(hh, [DirName filesep fbase '_' int2str(pg)], nodisplay, gfmt);
end

end


