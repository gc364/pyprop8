This directory contains scripts that aim to reproduce the figures and examples from [O'Toole & Woodhouse (2011)](https://doi.org/10.1111/j.1365-246X.2011.05210.x) and [O'Toole et al. (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x). In addition to `pyprop8`, these scripts require `matplotlib` to be installed.

Some minor differences can be seen between some of the output here, and the figures within the papers. These are likely due to differences in the way the source time functions are implemented and applied; unfortunately, the papers do not fully describe this step.

The `otoole_valentine_woodhouse_2012.py` file has comparisons of the `pyprop8` calculated excitation kernels for the moment tensor components
with the ones found using autodiff. These are also comparable to the figures in [O'Toole et al. (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x)
and serves to ensure that the autodiff gradients are accumulating correctly.