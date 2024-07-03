import pickle
import numpy as np

gate_dict = {
    "X":np.array([[0,1],[1,0]]),
    "Y":np.array([[0,-1j],[1j,0]]),
    "Z":np.array([[1,0],[0,-1]]),
    "H":np.array([[1,1],[1,-1]]) / np.sqrt(2),
    "T":np.array([[1,0],[0,np.exp(1j * np.pi / 4)]]),
    "S":np.array([[1,0],[0,1j]]),
    "CNOT":np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),
    "CZ":np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]]),
    "SWAP":np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]),
    "iSWAP":np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]),
}
"""
For a key, the respective value contains the gate as numpy array.
"""

coolable = {
    ("CNOT","H","Z"):False,
    ("CNOT","H","X"):False,
    ("CNOT","H","S"):False,
    ("CNOT","H","T"):True,
    ("iSWAP","H","Z"):False,
    ("iSWAP","H","X"):False,
    ("iSWAP","H","S"):False,
    ("iSWAP","H","T"):True,
}
"""
Whether the gate set (key) can be cooled or not.
"""

def gateset_string(gate_set:tuple) -> str:
    """
    Converts the gateset from the dataset dictionaries into a string that is nice for printing
    """
    gate_str = "{" + gate_set[0]
    for i in range(1,len(gate_set)):
        gate_str += ("," + gate_set[i])
    gate_str += "}"
    return gate_str

def read_file(fname:str) -> tuple:
    data = ()
    with open(fname,mode="rb") as res_file:
        while True:
            try:
                sim_data = pickle.load(res_file)
                data += (sim_data,)

            except EOFError:
                print("File " + fname + " is over.")
                break
    return data