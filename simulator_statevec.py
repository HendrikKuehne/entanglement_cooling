import numpy as np
import pickle
import json
from itertools import product
from datetime import datetime
from lib import ent_heating_statevec,MPS,ent_cooling_statevec

if __name__ == "__main__":
    # gates in the gate sets
    gate_dict = {
        "X":np.array([[0,1],[1,0]]),
        "H":np.array([[1,1],[1,-1]]) / np.sqrt(2),
        "T":np.array([[1,0],[0,np.exp(1j * np.pi / 4)]]),
        "S":np.array([[1,0],[0,1j]]),
        "CNOT":np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),
    }

    # physical parameters
    nWires = 4
    # beta = 5

    # simulation parameters
    steps_heat = 250
    steps_cool = 6000
    realizations = 100

    # file parameters
    now = datetime.now()
    timestr = now.strftime("%m-%d_%H-%M-%S")
    fname = timestr

    # defining the arguments
    thermalization_kwargs = [
        {
            "method":"statevec",
            "realizations":realizations,
            "heating":{
                "nWires":nWires,
                "gate_set":gate_set,
                "nSteps":steps_heat,
            },
            "cooling":{
                "gate_set":gate_set,
                "nSteps":steps_cool,
                "beta":beta,
            }
        } for gate_set,beta in product([("CNOT","H","X"),("CNOT","H","S"),("CNOT","H","T")],[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])]
    # [("CNOT","H","X"),("CNOT","H","S"),("CNOT","H","T")]

    # saving the configuration
    with open("data/" + fname + "___config.json","w") as jfile: json.dump(thermalization_kwargs,jfile)

    # creating a file for pickling the data
    pfile_name = "data/" + fname + "___data.pickle"
    with open(pfile_name,"w") as pfile: pass

    # code that simulates thermalization for the given kwargs
    for iKw,kwargs in enumerate(thermalization_kwargs):
        print(f"----- set {iKw+1} / {len(thermalization_kwargs)} of kwargs -----")

        # gate names to gates
        kwargs["gate_set"] = kwargs["heating"]["gate_set"]
        kwargs["heating"]["gate_set"] = [gate_dict[gate] for gate in kwargs["heating"]["gate_set"]]
        kwargs["cooling"]["gate_set"] = [gate_dict[gate] for gate in kwargs["cooling"]["gate_set"]]

        Svn_heating = np.zeros(shape=(kwargs["realizations"],kwargs["heating"]["nSteps"]+1,kwargs["heating"]["nWires"]-1))
        Svn_cooling = np.zeros(shape=(kwargs["realizations"],kwargs["cooling"]["nSteps"]+1,kwargs["heating"]["nWires"]-1))

        psi_heating = np.zeros(shape=(kwargs["realizations"],2**kwargs["heating"]["nWires"]),dtype=np.complex128)
        psi_cooling = np.zeros(shape=(kwargs["realizations"],2**kwargs["heating"]["nWires"]),dtype=np.complex128)

        # simulating the thermalization procedure for many realizations
        iR = 0
        while iR < kwargs["realizations"]:
            try:
                # heating
                psi_heating_,return_dict = ent_heating_statevec(**kwargs["heating"],return_Svn=True)
                psi_heating[iR,:] = psi_heating_
                Svn_heating[iR,:,:] = return_dict["Svn"]

                # cooling
                psi_cooling_,return_dict = ent_cooling_statevec(**kwargs["cooling"],state=psi_heating_.copy())
                psi_cooling[iR,:] = psi_cooling_
                Svn_cooling[iR,:,:] = return_dict["Svn"]

                iR += 1
            except np.linalg.LinAlgError:
                print(f"    iR = {iR}: LinAlgError; trying again.")
                continue

            if iR % 20 == 0: print("    iR = {} of {}".format(iR,kwargs["realizations"]))

        # saving the data
        kwargs["cooling"]["state"] = psi_cooling
        kwargs["psi_heating"] = psi_heating
        kwargs["psi_cooling"] = psi_cooling
        kwargs["Svn_heating"] = Svn_heating
        kwargs["Svn_cooling"] = Svn_cooling

        with open(pfile_name,"ab") as pfile:
            pickle.dump(kwargs,pfile)