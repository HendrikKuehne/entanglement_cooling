"""
The entanglement heating and cooling algorithms from "Irreversibility and Entanglement
Spectrum Statistics in Quantum Circuits", Shaffer et Al, 2014.
"""

import numpy as np
from lib.mps import MPS,init_spinup_MPS
from lib.utils import vn_entropy,apply_gate_statevec

def ent_heating_MPS(nWires:int,gate_set:tuple,nSteps:int=500,eps:float=1e-12,chi_max:int=None) -> tuple[MPS,dict]:
    """
    Entanglement heating of a random product state, using the gates in `gate_set`. Algorithm from "Irreversibility and Entanglement
    Spectrum Statistics in Quantum Circuits", Shaffer et Al, 2014. Simulated using MPS.
    
    Returns the resulting state and the entanglement entropy at every bond for every iteration.
    """
    state = init_spinup_MPS(L=nWires)

    # rotating each qubit to create the initial state from the paper
    RY = lambda phi: np.array([[np.cos(phi/2),(-1)*np.sin(phi/2)],[np.sin(phi/2),np.cos(phi/2)]])
    for iWire,theta in enumerate(np.random.uniform(low=0,high=np.pi,size=(nWires,))):
        state.apply_operator(RY(theta),iWire)

    # defining the random sequence of gates we are going to apply
    gate_indices = np.random.randint(low=0,high=len(gate_set),size=(nSteps))
    """Defines the gate that is applied in each step."""
    wire_indices = [tuple(np.random.choice(a=nWires,size=(int(np.log2(gate_set[iGate].shape[0])),),replace=False)) for iGate in gate_indices]
    """Defines the wires the respective gates are applied to."""

    return_vals = {"Svn":np.zeros(shape=(nSteps+1,state.L-1))}

    for iStep,iGate in enumerate(gate_indices):
        wires = wire_indices[iStep]
        state.apply_operator(gate_set[iGate],*wires,eps=eps,chi_max=chi_max)

        return_vals["Svn"][iStep+1,:] = state.entanglement_entropy()

    return state,return_vals

def ent_cooling_MPS(state:MPS,gate_set:tuple,beta:float,nSteps:int=500,eps:float=1e-12,chi_max:int=None) -> tuple[MPS,dict]:
    """
    Entanglement colling of the given state, using the gates in `gate_set`. Algorithm from "Irreversibility and Entanglement
    Spectrum Statistics in Quantum Circuits", Shaffer et Al, 2014.
    
    Returns the resulting state and the half-chain entanglement
    entropy for every iteration.
    """
    nWires = state.L

    # defining the random sequence of gates we are going to apply
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
        newstate.apply_operator(gate_set[iGate],*wires,eps=eps,chi_max=chi_max)

        Svn_new = newstate.entanglement_entropy().mean()
        # accepting the new state only with a probability, reminiscient of thermalization
        if Svn_new <= return_vals["Svn"][iStep,:].mean():
            # the new state has lower entropy; we'll take it
            state = newstate

        elif np.exp(beta * (return_vals["Svn"][iStep,:].mean() - Svn_new)) <= np.random.randint(low=0,high=1):
            # we'll take the new state due to thermalization
            state = newstate

        return_vals["Svn"][iStep+1,:] = state.entanglement_entropy()

    return state,return_vals

def ent_heating_statevec(nWires:int,gate_set:tuple,nSteps:int=500,return_Svn:bool=False) -> tuple[np.ndarray,dict]:
    """
    Entanglement heating of a random product state, using the gates in `gate_set`. Algorithm from "Irreversibility
    and Entanglement Spectrum Statistics in Quantum Circuits", Shaffer et Al, 2014.
    
    Returns the resulting state. If `return_Svn == True`, the entanglement entropy for cuts along every bond
    dimension is returned, additionally.
    """
    # initial state
    state = np.zeros(shape=(2**nWires,))
    state[0] = 1
    state = np.reshape(state,newshape=[2 for iWire in range(nWires)])

    # defining the random sequence of gates we are going to apply
    gate_indices = np.random.randint(low=0,high=len(gate_set),size=(nSteps))
    """Defines the gate that is applied in each step."""
    wire_indices = [tuple(np.random.choice(a=nWires,size=(int(np.log2(gate_set[iGate].shape[0])),),replace=False)) for iGate in gate_indices]
    """Defines the wires the respective gates are applied to."""

    return_vals = {"Svn":np.zeros(shape=(nSteps+1,nWires-1))}
    if return_Svn: return_vals["Svn"][0,:] = np.zeros(shape=(nWires-1,))

    for iStep,iGate in enumerate(gate_indices):
        wires = wire_indices[iStep]

        state = apply_gate_statevec(state,gate_set[iGate],wires)

        if return_Svn:
            # calculating the von-Neumann entropy
            return_vals["Svn"][iStep+1,:] = vn_entropy(state.flatten())

    if return_Svn:
        return state.flatten(),return_vals
    else:
        return state.flatten()

def ent_cooling_statevec(state:np.ndarray,gate_set:tuple,beta:float,nSteps:int=500) -> tuple[np.ndarray,dict]:
    """
    Entanglement cooling of a random product state, using the gates in `gate_set`. Algorithm from "Irreversibility and Entanglement
    Spectrum Statistics in Quantum Circuits", Shaffer et Al, 2014.

    The initial state `state` can be given as a statevector or as a tensor with shape `(2,2,...,2)`.

    Returns the resulting state and the entanglement entropy for cuts along every bond dimension.
    """
    # initial state
    if len(state.shape) == 1:
        # sanity check
        assert int(np.log2(len(state))) == np.log2(len(state)), "State must consist of qubits!"
        nWires = int(np.log2(len(state)))
        state = np.reshape(state,newshape=[2 for iWire in range(nWires)])
    else:
        assert all([dim == 2 for dim in state.shape]), "State must consist of qubits!"
        nWires = len(state.shape)

    # defining the random sequence of gates we are going to apply
    gate_indices = np.random.randint(low=0,high=len(gate_set),size=(nSteps))
    """Defines the gate that is applied in each step."""
    wire_indices = [tuple(np.random.choice(a=nWires,size=(int(np.log2(gate_set[iGate].shape[0])),),replace=False)) for iGate in gate_indices]
    """Defines the wires the respective gates are applied to."""

    return_vals = {"Svn":np.zeros(shape=(nSteps+1,nWires-1))}
    return_vals["Svn"][0,:] = vn_entropy(state.flatten())

    for iStep,iGate in enumerate(gate_indices):
        wires = wire_indices[iStep]

        # calculating the new state
        newstate = apply_gate_statevec(state.copy(),gate_set[iGate],wires)

        Svn_new = vn_entropy(newstate.flatten())
        # accepting the new state only with a probability, reminiscient of thermalization
        if Svn_new.mean() <= return_vals["Svn"][iStep,:].mean():
            # the new state has lower entropy; we'll take it
            state = newstate
            return_vals["Svn"][iStep+1,:] = Svn_new

        elif np.exp(beta * (return_vals["Svn"][iStep,:].mean() - Svn_new.mean())) <= np.random.randint(low=0,high=1):
            # we'll take the new state due to thermalization
            state = newstate
            return_vals["Svn"][iStep+1,:] = Svn_new

        else:
            # we reject the state
            return_vals["Svn"][iStep+1,:] = return_vals["Svn"][iStep,:].copy()

    return state.flatten(),return_vals
