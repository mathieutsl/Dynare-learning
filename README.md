# Learning expectations for Dynare

Social learning and constant-gain adaptive learning as alternatives to rational
expectations, packaged as a `+learning` MATLAB namespace plus two hooks in the
Dynare core code.

**Status.** This is not part of the official version of Dynare and only a working repo. The reference version is the merge request against Dynare master.  Once it is merged, use Dynare
directly — this repository will not be updated.

## What it does

- **Social learning**, following the toolbox of Grimaud, Salle and Vermandel
  (2025). A population of agents holds heterogeneous beliefs about the
  forward-looking variables; beliefs mutate, are ranked by a discounted
  forecast-error criterion, and spread through a pairwise tournament.
  Simulation, Monte Carlo moments, generalised IRFs, historical decomposition,
  and Bayesian estimation through an inversion filter.
- **Adaptive learning**, the constant-gain variant of Slobodyan and Wouters
  (2012). Simulation and moments only, no likelihood.

Configuration goes through naming conventions on parameters declared in the
`.mod` file; the preprocessor is not modified.

## Requirements

MATLAB. The package does not run under Octave: the belief and shock streams are
built as `RandStream` substreams, which have no Octave equivalent.

Developed and tested against Dynare `dev-7.x`.

## Installation

1. Copy `matlab/+learning/` into the `matlab/` directory of your Dynare
   installation.

2. Apply the two dispatch points to the core:

       cd <dynare>
       git apply /path/to/core_hooks.patch

   If your Dynare is not a git checkout, the patch only inserts blocks and
   removes nothing, so it can be applied by hand. `reference/` holds the two
   modified files for reading. It's better to not copy them into your installation, they
   are tied to one Dynare version and would silently shadow any later fix.

3. Check the installation by running the examples.

## Examples

Run them from `examples/`.

- `mod_example_simul_SL.mod` — social learning, simulation. Also writes
  `simdata_SL.mat`, the dataset used by the estimation example.
- `mod_example_estim_SL.mod` — social learning, Bayesian estimation. Runs in
  about a minute: the mode and its covariance are reloaded from
  `mod_example_estim_SL_reference_mode.mat`, which took several hours to obtain
  with `mode_compute=6`.
- `mod_example_AL.mod` — adaptive learning, simulation.

## Documentation

`doc/learning.pdf` describes the mechanisms, the option and output structures,
the implementation conventions that are not visible from the code, and the
configurations that are rejected. Source in `doc/learning.tex`.

## References

Grimaud, A., Salle, I. and Vermandel, G. (2025), "A Dynare toolbox for social
learning expectations", *Journal of Economic Dynamics and Control*, 172,
104984. https://doi.org/10.1016/j.jedc.2024.104984

The companion code: "Code for a Dynare toolbox for social learning
expectations", Mendeley Data. https://doi.org/10.17632/fzmx3vkt66.1

Slobodyan, S. and Wouters, R. (2012), "Learning in a medium-scale DSGE model
with expectations based on small forecasting models", *American Economic
Journal: Macroeconomics*, 4(2), 65–101.

## License

GNU General Public License v3.0 or later, same as Dynare. See `COPYING`.
Derived from Dynare (https://www.dynare.org), Copyright © Dynare Team.
The two files in `reference/` are Dynare core files modified to add the
dispatch points; the modifications are those of `core_hooks.patch`.

## Author

Written by Mathieu Tassel during a research internship at CEPREMAP (2026),
under the supervision of Gauthier Vermandel, Alex Grimaud and Isabelle Salle.