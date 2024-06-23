"""
The entanglement heating and cooling algorithms from "Irreversibility and Entanglement
Spectrum Statistics in Quantum Circuits", Shaffer et Al, 2014.
"""

import numpy as np
from lib.mps import MPS,init_spinup_MPS
import pennylane as qml

def renyi_entropy(p:np.ndarray,q:float) -> float:
    """Computing the Renyi-entropy of order q. All dimensions in p except the last are batch dimensions."""
    assert q > 0
    return np.log2(np.sum(p,axis=-1)**q) / (1-q)

def ent_heating(nWires:int,gate_set:tuple,nSteps:int=500) -> tuple[MPS,dict]:
    """
    Entanglement heating of a random product state, using the gates in `gate_set`. Algorithm from "Irreversibility and Entanglement
    Spectrum Statistics in Quantum Circuits", Shaffer et Al, 2014.
    
    Returns the resulting state and the half-chain entanglement
    entropy for every iteration.
    """
    state = init_spinup_MPS(L=nWires)

    # rotating each qubit to create the initial state from the paper
    RY = lambda phi: np.array([[np.cos(phi/2),(-1)*np.sin(phi/2)],[np.sin(phi/2),np.cos(phi/2)]])
    for iWire,theta in enumerate(np.random.uniform(low=0,high=np.pi,size=(nWires,))):
        state.apply_operator(RY(theta),iWire)

    gate_indices = np.random.randint(low=0,high=len(gate_set),size=(nSteps))
    """Defines the gate that is applied in each step."""
    wire_indices = [tuple(np.random.choice(a=nWires,size=(int(np.log2(gate_set[iGate].shape[0])),),replace=False)) for iGate in gate_indices]
    """Defines the wires the respective gates are applied to."""

    return_vals = {"Svn":np.zeros(shape=(nSteps+1,state.L-1))}

    for iStep,iGate in enumerate(gate_indices):
        wires = wire_indices[iStep]
        state.apply_operator(gate_set[iGate],*wires)

        return_vals["Svn"][iStep+1,:] = state.entanglement_entropy()

    return state,return_vals

def ent_cooling(state:MPS,gate_set:tuple,beta:float,nSteps:int=500) -> tuple[MPS,dict]:
    """
    Entanglement colling of the given state, using the gates in `gate_set`. Algorithm from "Irreversibility and Entanglement
    Spectrum Statistics in Quantum Circuits", Shaffer et Al, 2014.
    
    Returns the resulting state and the half-chain entanglement
    entropy for every iteration.
    """
    nWires = state.L

    gate_indices = np.random.randint(low=0,high=len(gate_set),size=(nSteps))
    """Defines the gate that is applied in each step."""
    wire_indices = [tuple(np.random.choice(a=nWires,size=(int(np.log2(gate_set[iGate].shape[0])),),replace=False)) for iGate in gate_indices]
    """Defines the wires the respective gates are applied to."""

    return_vals = {"Svn":np.zeros(shape=(nSteps+1,nWires-1))}
    return_vals["Svn"][0,:] = state.entanglement_entropy()

    for iStep,iGate in enumerate(gate_indices):
        wires = wire_indices[iStep]
        newstate = state.copy()
        # applying the gate to a test-state
        newstate.apply_operator(gate_set[iGate],*wires)

        Svn_new = newstate.entanglement_entropy().mean()
        # accepting the new state only with a probability, reminiscient of thermalization
        if Svn_new <= return_vals["Svn"][iStep,:].mean():
            state = newstate.copy()
            # print("Svn_new = {:.3f} <= {:.3f} = Svn_old".format(Svn_new,return_vals["Svn"][iStep]))
        else:
            # print("exp[-beta(Snew-Sold)] = exp[-beta({:.3f} - {:.3f})] = {:.3f}".format(Svn_new,return_vals["Svn"][iStep,:].mean(),np.exp((-1) * beta * (Svn_new - return_vals["Svn"][iStep,:].mean()))))
            if np.exp(beta * (return_vals["Svn"][iStep,:].mean() - Svn_new)) <= np.random.randint(low=0,high=1):
                state = newstate

        return_vals["Svn"][iStep+1,:] = state.entanglement_entropy()

    return state,return_vals