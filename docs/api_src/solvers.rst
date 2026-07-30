``solvers``
============

FFT-based variational solvers, grouped by problem type.

Shared types
------------

.. automodule:: solvers.types
   :members:
   :show-inheritance:

``solvers.mechanical``
-----------------------

Strain- and displacement-based Newton-CG solvers.

.. automodule:: solvers.mechanical.strain_nw_cg
   :members:
   :show-inheritance:

.. automodule:: solvers.mechanical.displacement_nw_cg
   :members:
   :show-inheritance:

``solvers.damage``
--------------------

Phase-field damage (Helmholtz CG) and staggered-loop accelerators.

.. automodule:: solvers.damage.pff_damage
   :members:
   :show-inheritance:

.. automodule:: solvers.damage.anderson
   :members:
   :show-inheritance:
