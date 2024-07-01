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
    "singvals_heating":np.ndarray,
    "singvals_cooling":np.ndarray,
}
``

Do watch out, because there is a misnomer here: The dictionaries in `*.pickle` contain keys `singvals_heating` and `singvals_cooling`, these are not the singular values however! These actually contain the entanglement entropies for bipartitions $2^N\mapsto 2^{N_A}\otimes 2^{N-N_A}$.

## Datasets

* Simulated without initial randomized rotation during heating:
    * `06-29_01-59-54`: Gate sets $`\{\text{CNOT},H,X\}`$, $`\{\text{CNOT},H,S\}`$, $`\{\text{CNOT},H,T\}`$ for $\beta\in [1,10]$
    * Simulating different numbers of wires:
        * `06-29_01-54-34`: $`N\in\{4,5,6,7,8,9,10,11,12\}`$, gate set $`\{\text{CNOT},H,X\}`$
        * `06-29_01-55-08`: $`N\in\{4,5,6,7,8,9,10,11,12\}`$, gate set $`\{\text{CNOT},H,S\}`$
        * `06-29_01-55-15`: $`N\in\{4,5,6,7,8,9,10,11,12\}`$, gate set $`\{\text{CNOT},H,T\}`$

## A few caveats