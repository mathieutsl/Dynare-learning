/*
 * Constant-gain adaptive learning in a small New Keynesian model.
 *
 * Agents do not know the model's law of motion. They forecast the forward
 * variables with a small perceived law of motion (PLM) whose coefficients are
 * updated every period by constant-gain recursive least squares. The actual law
 * of motion is then re-solved from those beliefs, so the transition matrices
 * move over time. A projection facility keeps the last stable solution whenever
 * a belief update would make the model diverge.
 *
 * CONFIGURATION
 *
 *     AL_active      1 turns adaptive learning on, 0 falls back to the
 *                    rational-expectations path
 *     AL_gain        constant gain of the RLS updating. Larger means beliefs
 *                    react faster to recent forecast errors and forget the past
 *                    sooner;
 *     AL_pj_threshold Projection facility threshold to keep the model solution stable
 *     AL_plm_type    perceived law of motion: 0 for intercept only, 1 for AR(1)
 *     AL_learn_<var> to declared which variable are leanred
 *
 * OUTPUT
 *   oo_.endo_simul                             simulated path
 *   oo_.learning.adaptive.info.beta_hist       [nfwrd x nb x T] belief path
 *   oo_.learning.adaptive.info.beta_final      terminal beliefs
 *   oo_.learning.adaptive.info.pf_hits         number of projection facility hits
 *   oo_.learning.adaptive.moments              MC moments with empirical bands
 *
 */

var s_d s_s s_i i y pi tau dy;

varexo  e_d   ${\varepsilon_{D}}$  (long_name='Demand shock')
        e_s   ${\varepsilon_{S}}$  (long_name='Supply shock')
        e_i   ${\varepsilon_{I}}$  (long_name='Monetary policy shock');


parameters
        rho_s   ${\rho_{S}}$    (long_name='AR(1) supply')
        rho_d   ${\rho_{D}}$    (long_name='AR(1) demand')
        rho_i   ${\rho_{I}}$    (long_name='AR(1) monetary')
        beta    ${\beta}$       (long_name='Discount factor')
        kappa   ${\kappa}$      (long_name='Slope of the Phillips curve')
        phi_pi  ${\phi_{\pi}}$  (long_name='Inflation stance')
        phi_y   ${\phi_{y}}$    (long_name='Output stance')
        sigma   ${\sigma}$      (long_name='Risk aversion')
        rho_mp  ${\rho_{MP}}$   (long_name='Policy rate smoothing')

        // Adaptive learning
        AL_active       ${\mathbb{1}_{AL}}$ (long_name='AL master switch')
        AL_learn_pi     ${\ell_{\pi}}$      (long_name='AL learns pi')
        AL_learn_y      ${\ell_{y}}$        (long_name='AL learns y')
        AL_gain         ${\gamma}$          (long_name='AL constant gain')
        AL_plm_type     ${\mathrm{PLM}}$    (long_name='AL perceived law of motion')
        AL_pj_threshold ${\bar{\rho}}$      (long_name='AL projection facility threshold');

rho_d  = 0.50;
rho_s  = 0.30;
rho_i  = 0.00;
rho_mp = 0.90;
beta    = 0.99;
kappa   = 0.03;
sigma   = 1.00;
phi_pi  = 1.50;
phi_y   = 0.125;

AL_active = 1;
AL_gain   = 0.01;
AL_plm_type = 1; // AR(1) beliefs; set to 0 for intercept-only
AL_learn_y= 1;    
AL_learn_pi =1 ;
AL_pj_threshold = 1;   // Set it to 0.868 to see the facility bind. Under this
                       // calibration max|eig(T)| is in [0.847, 0.875] as
                       // beliefs drift, this threshold value rejects about 3.5% of the updates. It is a
                       // numerical safeguard, not a structural parameter, and
                       // it is tied to this calibration. Note that the facility
                       // can only bind with AL_plm_type = 1: with an
                       // intercept-only PLM beliefs carry no slope, T does not
                       // depend on them, and the spectral radius is constant.



model(linear);
    // Phillips curve
    pi = beta*pi(+1) + kappa*y + s_s;

    // Euler equation
    y = y(+1) - (1/sigma)*(i - pi) + s_d;

    // Taylor rule
    i = rho_mp*i(-1) + (1-rho_mp)*(phi_pi*tau + phi_y*y) + s_i;

    // Inflation target of the policy rule
    tau = pi;

    // Exogenous processes
    s_d = rho_d*s_d(-1) + e_d;
    s_s = rho_s*s_s(-1) + e_s;
    s_i = rho_i*s_i(-1) + e_i;

    // Growth observable
    dy = y;
end;

shocks;
    var e_d = 0.10;
    var e_s = 0.05;
    var e_i = 0.01;
end;

steady;
check;

// Current helper being shared with social learning require using a RNG that support substream
set_dynare_seed('mrg32k3a', 1);

// irf is left at 0: adaptive learning IRFs are not implemented and the fallback
// would report rational-expectations responses
stoch_simul(order=1, periods=1000, drop=200, replic=100, irf=0);
