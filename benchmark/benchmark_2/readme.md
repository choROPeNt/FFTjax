# Random Periodic RVE Benchmark Data

`assets/load_*_phi_*.csv` (and the earlier digitizations in `assets/archiv/`)
contain reference data from:

> Varandas, L. F., Catalanotti, G., Melro, A. R., & Falzon, B. G. (2020).
> *On the importance of nesting considerations for accurate computational
> damage modelling in 2D woven composite materials.*
> Computational Materials Science, 172, 109323.
> https://doi.org/10.1016/j.commatsci.2019.109323

Reference RVEs at fibre volume fractions phi = 0.35 / 0.55 / 0.75, under
periodic boundary conditions, compared against tension/compression along the
loaded axis (load_11/22) and in-plane shear (load_12/13/23).

`load_22_tension_phi_0.XX_.csv`, `load_22_comp_phi_0.XX_.csv`,
`load_13_shear_phi_0.XX_.csv`, `load_23_shear_phi_0.XX_.csv`: stress-strain
curves digitized from the paper's figures, used as reference curves for
`pff_random_periodic.py`'s tension_x / compression_x / shear_xy load paths.

The random fibre-packing algorithm underlying `generation.rve.make_random_composite_rve`
(Catalanotti 2016) traces back to:

> Melro, A. R., Camanho, P. P., & Pinho, S. T. (2008).
> *Generation of random distribution of fibres in long-fibre reinforced composites.*
> Composites Science and Technology, 68(9), 2092-2102.
> https://doi.org/10.1016/j.compscitech.2008.03.013
