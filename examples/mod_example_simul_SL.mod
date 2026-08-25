/*
 * Social learning in a small New Keynesian model - SIMULATION example.
 *
 * Agents hold heterogeneous beliefs about the forward-looking variables. Each
 * period a share of them receives news, beliefs are compared through a
 * discounted forecast-error fitness criterion, and pairs of agents adopt the
 * fitter of the two. The aggregate belief wedge feeds back into the policy
 * function, so the model is no longer solved under rational expectations.
 *
 * For the estimation interface, see mod_example_estim_SL.mod. The two files
 * declare the same model and the same social-learning setup; only the final
 * command differs.
 *
 * CONFIGURATION
 *
 *   Social learning is configured through parameters, not through options of
 *   the stoch_simul command.
 *
 *     SL_active                  1 turns social learning on. It must be set
 *                                explicitly: declaring SL_* parameters is not
 *                                enough. 0 or absent falls back to the
 *                                rational-expectations path, for estimation as
 *                                well as for simulation.
 *     SL_learn_<var>             1 marks the forward variable <var> as learned,
 *                                0 keeps it at rational expectations. A forward
 *                                variable with no flag is not learned. This is
 *                                the only thing that selects the learned set.
 *
 *   Each learned variable requires all three of the following. None has a
 *   default value.
 *
 *     SL_mutation_std_<var>      belief dispersion xi. 
 *     SL_fitness_discount_<var>  fitness memory delta, in (0,1)
 *     SL_mutation_prob_<var>     share of agents receiving news each period,
 *                                in (0,1]
 *
 *   The global options do have defaults:
 *
 *     SL_n_agents                population size J                      [300]
 *     SL_wedding_proportion      share entering the tournament          [1.0]
 *     SL_t_max                   horizon of the reported h-step-ahead
 *                                expectations                            [1]
 *
 *   Every learned variable needs a matching a_<var> shock declared in varexo.
 *
 * THE VARIANCE OF a_<var>
 *
 *   a_<var> carries two things. It is the channel through which aggregate news
 *   reaches the population, and it is where the resolved belief wedge is
 *   stored once the tournament has run. The wedge itself is never drawn; only
 *   the news is.
 *
 *   In simulation the variance is a modelling choice, not a requirement:
 *     - zero variance is pure social learning. Beliefs deviate from rational
 *       expectations only through what the mechanism generates on its own.
 *     - a positive variance adds an exogenous news shock on top. Unlike an
 *       ordinary expectation shock it does not just move expectations for one
 *       period: it enters the beliefs of the agents who receive it, and it survives
 *       in more periods through the learning process.
 *
 *   This file uses a positive variance, because the estimation example runs on
 *   the data it produces and a zero-variance shock would make the likelihood
 *   degenerate.
 *
 * DATA FOR THE ESTIMATION EXAMPLE
 *
 *   simdata_SL.mat is produced from this file by saving oo_.endo_simul with
 *   drop = 0 and replic = 1.
 */

var s_d s_s s_i i y pi tau dy pie;

varexo  e_d   ${\varepsilon_{D}}$  (long_name='Demand shock'),
        e_s   ${\varepsilon_{S}}$  (long_name='Supply shock'),
        e_i   ${\varepsilon_{I}}$  (long_name='Monetary policy shock'),
        a_pi  ${a_{\pi}}$          (long_name='Aggregate belief wedge on pi');

parameters
        rho_s   ${\rho_{S}}$    (long_name='AR(1) supply'),
        rho_d   ${\rho_{D}}$    (long_name='AR(1) demand'),
        rho_i   ${\rho_{I}}$    (long_name='AR(1) monetary'),
        beta    ${\beta}$       (long_name='Discount factor'),
        kappa   ${\kappa}$      (long_name='Slope of the Phillips curve'),
        phi_pi  ${\phi_{\pi}}$  (long_name='Inflation stance'),
        phi_y   ${\phi_{y}}$    (long_name='Output stance'),
        sigma   ${\sigma}$      (long_name='Risk aversion'),
        rho_mp  ${\rho_{MP}}$   (long_name='Policy rate smoothing'),

        // Social learning
        SL_active              ${\mathbb{1}_{SL}}$ (long_name='SL master switch'),
        SL_learn_pi            ${\ell_{\pi}}$      (long_name='SL learns pi'),
        SL_mutation_std_pi     ${\xi_{\pi}}$       (long_name='SL belief dispersion, pi'),
        SL_fitness_discount_pi ${\delta_{\pi}}$    (long_name='SL fitness memory, pi'),
        SL_mutation_prob_pi    ${\mu_{\pi}}$       (long_name='SL news probability, pi'),
        SL_n_agents            ${J}$               (long_name='SL population size'),
        SL_wedding_proportion  ${\omega}$          (long_name='SL tournament share'),
        SL_t_max               ${h}$               (long_name='SL expectation horizon');

rho_d   = 0.80;
rho_s   = 0.50;
rho_i   = 0.00;
beta    = 0.99;
kappa   = 0.03;
sigma   = 1.00;
phi_pi  = 1.50;
phi_y   = 0.125;
rho_mp  = 0.00;

SL_active              = 1;
SL_n_agents            = 1000;
SL_wedding_proportion  = 1;
SL_t_max               = 3;

SL_learn_pi            = 1;
SL_mutation_std_pi     = 0.003;
SL_fitness_discount_pi = 0.50;
SL_mutation_prob_pi    = 0.4;


model(linear);

    // Subjective expectation of each learned variable: the rational value plus
    // the aggregate belief wedge. 
    pie = pi(+1) + a_pi;

    // Phillips curve with pie, not pi(+1)
    pi = beta*pie + kappa*y + s_s;

    // Euler equation with pie instead of pi(+1)
    y = y(+1) - (1/sigma)*(i - pie) + s_d;

    // Taylor rule
    i = rho_mp*i(-1) + (1-rho_mp)*(phi_pi*tau + phi_y*y) + s_i;

    // Inflation target of the policy rule
    tau = pi;

    // Growth observable
    dy = y;

    // Exogenous processes
    s_d = rho_d*s_d(-1) + e_d;
    s_s = rho_s*s_s(-1) + e_s;
    s_i = rho_i*s_i(-1) + e_i;

end;

shocks;
    var e_d = 0.10;
    var e_s = 0.05;
    var e_i = 0.01;
    var a_pi = 0.02^2;   // set to 0 for pure social learning; must be
                         // strictly positive to produce the estimation dataset
end;

steady;
check;

// Social learning requires an RNG algorithm that supports substreams, so that
// shocks and belief paths can be drawn independently. Declaring the seed here
// makes the run reproducible and silences the promotion notice.

set_dynare_seed('mrg32k3a', 1);

// Native stoch_simul options drive the social-learning path:
//   periods  length of each simulated path
//   drop     burn-in discarded before measuring moments
//   replic   Monte Carlo replications over belief draws, needed even at
//            order 1, since the trajectory depends on the belief draw
//   irf      horizon of the generalised IRFs

stoch_simul(order=1, periods=300, drop=0, replic=500, irf=20);

// Writes the dataset used by mod_example_estim_SL.mod. Comment out this block
// to use the file only for simulation.
verbatim;
    obs_names = {'dy','pi','i','pie'};
    d = struct();
    for k = 1:numel(obs_names)
        idx = find(strcmp(M_.endo_names, obs_names{k}), 1);
        d.(obs_names{k}) = oo_.endo_simul(idx, :).';
    end
    save('simdata_SL.mat', '-struct', 'd');
    fprintf('simdata_SL.mat written: %d obs x %d series\n', ...
            size(oo_.endo_simul, 2), numel(obs_names));
end;