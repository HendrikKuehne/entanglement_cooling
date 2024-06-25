# Contents

* `heating_cooling-py`: The entanglement heating- and cooling routines for both MPs and statevector.
* `mps.py`: The `MPS` class, and initialization of specific MPS.
* `utils.py`: Von-Neumann and Renyi-Entropy.

# ToDos:

* Optimize `utils/vn_entropy` by using consecutive SVDs instead of treating the whole system in every iteration (i.e. find the canonical form of the MPS corresponding to the statevector $\ket{\psi}$)