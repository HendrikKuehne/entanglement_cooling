import numpy as np
from scipy.linalg import svd

def renyi_entropy(p:np.ndarray,q:float) -> float:
    """Computing the Renyi-entropy of order q. All dimensions in p except the last are batch dimensions."""
    assert q > 0
    return np.log2(np.sum(p,axis=-1)**q) / (1-q)

def vn_entropy(statevector:np.ndarray) -> np.ndarray:
    """Von-Neumann entanglement entropy of the statevector `statevector`."""
    assert np.isclose(np.linalg.norm(statevector),1)
    nWires = int(np.log2(len(statevector)))
    Svn = ()

    for nSub in range(1,nWires):
        C = np.reshape(statevector.copy(),newshape=(2**(nSub),2**(nWires - nSub)))
        U,S,Vh = svd(C,full_matrices=False)

        # are we dealing with a valid set of singular values?
        assert np.isclose(np.linalg.norm(S),1)

        # discarding small singular values for numerical stability
        S = S[S > 1e-12]

        # calculating the entropy
        S2 = S * S
        Svn += (-np.sum(S2 * np.log2(S2)),)
    
    return np.array(Svn)

def apply_gate_statevec(state:np.ndarray,gate:np.ndarray,wires:tuple) -> np.ndarray:
    """Applies `gate` to `state`, where the state is a tensor with shape `(2,2,...,2)`."""
    # sanity check
    assert all([dim == 2 for dim in state.shape]), "State must consist of qubits!"
    nWires = len(state.shape)

    if len(wires) == 1:             # we got ourselves a single-qubit gate
        iWire = wires[0]
        # applying the gate
        state = np.tensordot(gate,state,axes=((1,),(iWire,)))

        # reshaping (tensordot appends non-contracted dimensions at the end)
        indices = np.concatenate((
            np.arange(iWire)+1,
            [0,],
            np.arange(iWire + 1,nWires)
        ))
        state = np.transpose(state,indices)
        return state

    if len(wires) == 2:             # we got ourselves a two-qubit gate
        i1,i2 = wires
        if i2 < i1:
            # since the indices are flipped, we also need to flip the gate
            SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            op = SWAP @ gate @ SWAP
            i1,i2 = i2,i1
        else:
            op = gate

        op = np.reshape(op,(2,2,2,2))

        # applying the gate
        state = np.tensordot(op,state,axes=((2,3),(i1,i2)))

        # reshaping (tensordot appends non-contracted dimensions at the end)
        indices = np.concatenate((
            np.arange(i1) + 2,
            [0,],
            np.arange(i2 - i1 - 1) + i1 + 2,
            [1,],
            np.arange(nWires-i2-1) + i2 + 1
        ))
        state = np.transpose(state,indices)
        return state

    return None

def bipartite_split(psi:np.ndarray) -> np.ndarray:
    """
    Calculates the singular values for a bipartite split of the state `psi`.
    State is assumed to be stored in the last dimension of `psi`.
    """
    # sanity check
    assert np.allclose(np.linalg.norm(psi,axis=-1),1)

    nWires = int(np.log2(psi.shape[-1]))
    nBip = int(nWires/2)
    C = np.reshape(psi.copy(),newshape=psi.shape[:-1] + (2**(nBip),2**(nWires - nBip)))
    U,S,Vh = np.linalg.svd(C,full_matrices=False)

    return S

def level_spacing_spectrum(ent:np.ndarray) -> np.ndarray:
    """
    Calculates the level spacing ratio statistic of the schmidt values given in `ent`.
    The first dimension of `ent` runs over the realizations, the second
    one runs over the singular values at the bonds.
    """
    if False:
        # averaging over the singular values
        ent_avg = ent.mean(axis=0)

        # sorting the singular values in descending order
        ent_avg = ent_avg[np.argsort(ent_avg)][::-1]

        spacings = ent_avg[:-1] - ent_avg[1:]

        ratios = spacings[1:] / spacings[:-1]
        return ratios
    
    # sorting the singular values in descending order
    print(np.argsort(ent,axis=1))
    ent = ent[np.argsort(ent,axis=1)][:,::-1]
    
    spacings = ent[:,-1] - ent[:,1:]
    ratios = spacings[:,1:] / spacings[:,:-1]

    # dropping nan or inf
    ratios = ratios[np.isfinite(ratios)]

    return np.array(ratios)
