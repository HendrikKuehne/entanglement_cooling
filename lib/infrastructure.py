import pickle

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