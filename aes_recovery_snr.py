from IPython.core.pylabtools import figsize

import dpa
import numpy as np
import matplotlib.pyplot as plt
import h5py



def SNR_Analysis(inputs, traces):

    len_traces = traces.shape[1]
    input_len = inputs.shape[0]


    print(f"[+] Producing SNR testing results")
    snr_highest = np.zeros((16, len_traces))
    for key_byte in range(16):
        snr_dist = np.zeros((256, len_traces))

        input_key_bytes = inputs[:, key_byte]

        # Precompute predicted leakage values
        precomputed_HW = np.zeros((256, input_len), dtype="int")
        for key_candidate in range(256):
            precomputed_HW[key_candidate, :] = dpa.HW[dpa.SubBytes[dpa.AddRoundKey(input_key_bytes, key_candidate)]]


        snr_dist = dpa.SNR_HW(9, precomputed_HW, traces)
        snr_highest[key_byte, :] = snr_dist[np.argmax(np.max(snr_dist, axis=1))]
        highest_time_point = np.argmax(snr_highest[key_byte, :])
        print(f"Highest SNR for key byte {key_byte} is {np.argmax(np.max(snr_dist, axis=0))}")
        print(f"Highest SNR time point for key byte {key_byte} is {np.argmax(np.max(snr_dist, axis=1))}")

    #
    # top_tenth_percentile_snr = np.argsort(-snr_highest, axis=1)[:, :int(0.1 * len_traces)]
    #

    fig, ax = plt.subplots(figsize=(15, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    ax.set_xlabel("time", fontsize=14)
    ax.set_ylabel("SNR", fontsize=14)
    for key_byte in range(16):
        row, col = key_byte // 4, key_byte % 4
        snr_data = snr_highest[key_byte, :]
        ax.plot(snr_data, label=f"State byte {key_byte}", color=colors[key_byte])

    ax.set_title("SNR Analysis for Each AES State Byte")
    plt.legend(loc="upper right", ncol=4, fontsize=10)
    fig.savefig("SNR_HW_Vectorized.png", dpi=300)