import dpa
import aes_recovery_snr, aes_recovery_dom, aes_recovery_correlation
import matplotlib.pyplot as plt
import h5py
import numpy as np

if __name__ == "__main__":
    # Load WS2.h5 dataset
    file = h5py.File("WS2.h5")
    dset = file["WS2"]
    inputs = np.array(dset[:, 0:16], dtype="int")
    traces = dset[:, 16:100016]
    file.close()

    aes_recovery_correlation.Correlation_Analysis(inputs, traces)
    aes_recovery_dom.DoM_Analysis(inputs, traces)
    aes_recovery_snr.SNR_Analysis(inputs, traces)
