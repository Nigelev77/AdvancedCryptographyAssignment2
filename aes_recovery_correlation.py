import numpy as np
import matplotlib.pyplot as plt
import dpa
import h5py


def Correlation_Analysis(inputs, traces):
    len_traces = traces.shape[1]
    print(f"[+] Producing Correlation testing results")

    # Perform correlation analysis to reduce number of time points required

    # Correlation testing

    pcor_rankings = dpa.Pcor_HW(inputs, traces)


    print(f"Top ranked keys")

    for key_byte in range(pcor_rankings.shape[0]):
        print(f"=========== Rankings for key byte {key_byte} =========")
        for key_candidate_idx in range(pcor_rankings.shape[1]):
            print(f"Key ranked {key_candidate_idx+1} is {pcor_rankings[key_byte, key_candidate_idx]}")

    print(f"Most likely key byte values are {" ".join(pcor_rankings[:, 0].astype(str))}")
    most_likely_key = "".join(f"{b:02x}" for b in pcor_rankings[:, 0])
    print(f"As a hex string that is: 0x{most_likely_key}")




