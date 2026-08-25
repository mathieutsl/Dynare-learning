/*
 * Social learning in a small New Keynesian model - ESTIMATION example.
 *
 * Agents hold heterogeneous beliefs about the forward-looking variables. Each
 * period a share of them receives news, beliefs are compared through a
 * discounted forecast-error fitness criterion, and pairs of agents adopt the
 * fitter of the two. The aggregate belief wedge feeds back into the policy
 * function, so the model is no longer solved under rational expectations.
 *
 * The model and the social-learning setup are those of
 * mod_example_simul_SL.mod, which also produces the dataset. This header only
 * documents what is specific to estimation.
 *
 * INVERSION FILTER
 *
 *   Estimation uses an inversion filter rather than the Kalman filter:
 *
 *   The inversion requires #observables == #shocks, the a_<var> news channels
 *   counting as shocks. Here: 3 structural shocks + 1 news channel = 4, hence 4
 *   observables.
 *
 * WHICH SL PARAMETERS ARE ESTIMATED
 *
 *   delta and mu are estimated, xi is calibrated in this example. The likelihood is
 *   is not continuous in xi: with a finite population, a small change in xi flips the
 *   ordering of winners in the tournament and the log-posterior jumps. To estimate it correctly a strategy 
 *  is to increase number of agents, try several optimization algorithm and starting point
 *  so for this example it's only calibrated.
 *
 * MODE AND HESSIAN
 *
 *   The numerical Hessian is not always trustable as some SL parameters landscape are 
 *   very noisy. The mode and its covariance were obtained with
 *   mode_compute=6, which samples around the mode and therefore returns a
 *   positive definite matrix. That run took about 6h30 with SL_n_agents = 1000.
 *   They are reloaded here from mode_file so that the example runs in a minute;
 *   re-run with mode_compute=6 to reproduce the mode file. The mode_check
 *   slices confirm that the point is a local maximum.
 */

var s_d s_s s_i i y pi tau dy pie ;

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
    y = y(+1)- (1/sigma)*(i - pie) + s_d;

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

// Social learning requires an RNG algorithm that supports substreams, so that
// shocks and belief paths can be drawn independently. Declaring the seed here makes
// the run reproducible and silences the promotion notice.
set_dynare_seed('mrg32k3a', 1);

// 4 observables for 4 shocks (e_d, e_s, e_i, a_y), as the inversion
// filter requires. pie stand for survey forecast series.
varobs dy pi i pie ;

estimated_params;
    // structural
    kappa,   0.03,  0.001, 0.50, gamma_pdf, 0.05,  0.03;
    phi_pi,  1.50,  1.001, 3.00, gamma_pdf, 1.50,  0.25;
    phi_y,   0.125, 0.001, 1.00, gamma_pdf, 0.125, 0.05;
    rho_d,   0.80,  0.01,  0.99, beta_pdf,  0.70,  0.15;
    rho_s,   0.50,  0.01,  0.99, beta_pdf,  0.50,  0.15;

    // structural shock standard deviations 
    stderr e_d,  0.3162, 0.001, 5, inv_gamma_pdf, 0.3162, 2;
    stderr e_s,  0.2236, 0.001, 5, inv_gamma_pdf, 0.2236, 2;
    stderr e_i,  0.1000, 0.001, 5, inv_gamma_pdf, 0.1000, 2;
    stderr a_pi,  0.02, 0.001, 5, inv_gamma_pdf, 0.02, 2;

    // social learning
    SL_fitness_discount_pi, 0.50,  0.01,   0.99, beta_pdf, 0.50,  0.10;
    SL_mutation_prob_pi,    0.4, 0.01, 0.99, beta_pdf, 0.4, 0.10;
end;



// The SL likelihood is not smooth: test several optmizer to find a good mode 
// before computing MCMC

estimation(order=1, datafile=simdata_SL, prefilter=1, mode_compute=0, mode_file=mod_example_estim_SL_reference_mode,
           mh_replic=0, mode_check);
