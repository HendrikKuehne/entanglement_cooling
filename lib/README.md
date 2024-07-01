# Contents

* `heating_cooling-py`: The entanglement heating- and cooling routines for both MPs and statevector.
* `mps.py`: The `MPS` class, and initialization of specific MPS.
* `utils.py`: Von-Neumann and Renyi-Entropy.

# ToDos:

* Optimize `utils/vn_entropy` by using consecutive SVDs instead of treating the whole system in every iteration (i.e. find the canonical form of the MPS corresponding to the statevector $\ket{\psi}$)
* Find a way to fix `MPS.apply_operator` for two-qubit-operators; I cant, for the life of me, get this to work such that the canonical form is conserved (i.e. such that I get valid entanglement entropies)