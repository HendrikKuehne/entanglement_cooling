"""The entanglement heating and cooling algorithms."""

import numpy as np
from lib.mps import MPS,init_spinup_MPS

def ent_heating(nWires:int,gate_set:tuple,nSteps:int=500) -> tuple[MPS,np.ndarray]:
    """
    Performs entanglement heating of a random product state, using the gates in `gate_set`. Returns the resulting state and the half-chain entanglement entropy for every iteration.
    """
    state = init_spinup_MPS(L=nWires)

    # rotating each qubit to create the initial state from the paper
    RY = lambda phi: np.array([[np.cos(phi/2),(-1)*np.sin(phi/2)],[np.sin(phi/2),np.cos(phi/2)]])
    for iWire,theta in enumerate(np.random.uniform(low=0,high=np.pi,size=(nWires,))):
        state.apply_operator(RY(theta),iWire)

    H = np.array([[1,1],[1,-1]]) / np.sqrt(2)
    T = np.array([[1,0],[0,np.exp(1j * np.pi / 4)]])
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    gate_set = (H,T,CNOT)

    gate_indices = np.random.randint(low=0,high=len(gate_set),size=(nSteps))
    wire_indices = [tuple(np.random.choice(a=nWires,size=(int(np.log2(gate_set[iGate].shape[0])),),replace=False)) for iGate in gate_indices]

    half_chain_entanglement = np.zeros(shape=(nSteps+1,))

    for iStep,iGate in enumerate(gate_indices):
        wires = wire_indices[iStep]
        state.apply_operator(gate_set[iGate],*wires)

        half_chain_entanglement[iStep+1] = state.entanglement_entropy()[int(nWires/2)]

    return state,half_chain_entanglement