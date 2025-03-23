import dpa
import numpy as np
import matplotlib.pyplot as plt
import h5py

def DoM_Analysis(inputs, traces):
    len_traces = traces.shape[1]
    input_len = inputs.shape[0]

    print(f"[+] Producing DoM distinguisher results")
    # DoM testing

    dom_rankings = dpa.DOM_Vectorized(inputs, traces)
    print(f"Top ranked keys")

    for key_byte in range(dom_rankings.shape[0]):
        print(f"=========== Rankings for key byte {key_byte} =========")
        for key_candidate_idx in range(dom_rankings.shape[1]):
            print(f"Key ranked {key_candidate_idx + 1} is {dom_rankings[key_byte, key_candidate_idx]}")

    print(f"Most likely key byte values are {" ".join(dom_rankings[:, 0].astype(str))}")
    most_likely_key = "".join(f"{b:02x}" for b in dom_rankings[:, 0])
    print(f"As a hex string that is: 0x{most_likely_key}")


if __name__ == "__main__":
    # Load WS2.h5 dataset
    file = h5py.File("WS2.h5")
    dset = file["WS2"]
    inputs = np.array(dset[:, 0:16], dtype="int")
    traces = dset[:, 16:100016]
    file.close()

    DoM_Analysis(inputs, traces)