# Overview of the data

## Format

Each file `*.pickle` contains a dictionary with the following structure:

``
{
    "heating":{
        "nWires":int,
        "gate_set":tuple[str],
        "nSteps":int,
    },
    "cooling":{
        "state":np.ndarray,
        "gate_set":tuple[str],
        "nSteps":int,
        "beta":float,
    }
    "gate_set":tuple[str],
    "psi_heating":np.ndarray,
    "psi_cooling":np.ndarray,
    "Svn_heating":np.ndarray,
    "Svn_cooling":np.ndarray,
}
``

* `psi_heating` / `psi_cooling`: The quantum states after the heating or cooling procedure, respectively. Two-dimensional `np.ndarray`, where the first dimension runs over the realizations and the second dimension contains the states themselves.
* `Svn_heating` / `Svn_cooling`: The von-Neumann entanglement entropies for bipartitions $2^N\mapsto 2^{N_A}\otimes 2^{N-N_A}$ that were recorded during heating or cooling, respectively. Three-dimensional `np.ndarray`, where the first dimension runs over the realizations, the second dimensions runs over the steps of the iterations and the third dimensions runs over the bipartitions.

Every file is named after date and time of the simulation, according to `MM-DD_HH_mm_SS`.

## Datasets

* Testing different values of $\beta$:
    * `07-02_19-44-30`: $`\beta\in [1,10]`$, gate set in $`\{\text{CNOT},H,X\},\{\text{CNOT},H,S\},\{\text{CNOT},H,T\}`$
* Simulating different numbers of wires:
    * `07-02_19-40-49`: $`N\in [4,11]`$, gate set $`\{\text{CNOT},H,X\}`$
    * `07-02_19-40-58`: $`N\in [4,11]`$, gate set $`\{\text{CNOT},H,S\}`$
    * `07-02_19-41-03`: $`N\in [4,11]`$, gate set $`\{\text{CNOT},H,T\}`$