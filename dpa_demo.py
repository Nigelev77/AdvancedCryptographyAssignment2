from dpa import *


if __name__ == '__main__':

    # decide what you wish to execute
    doDOM = 0
    doCorr = 0
    doSNR=1
    dosimMasking =0



    # load  dataset single SubBytes operation, 5000 points, 2000 traces
    # power traces and input bytes are stored in a HDF5 file, correct key byte is 119
    fhandle = h5py.File('AT89_SubBytes.h5')
    all_traces =  fhandle.get('traces')
    num_traces = 2000
    traces = all_traces[0:num_traces, :]
    len_traces = traces.shape[1]
    all_inputs = fhandle.get('inputs')
    inputs = all_inputs[0:num_traces]
    fhandle.close()

    # load  dataset single SubBytes operation, 5000 points, 200 traces
    # power traces and input bytes are stored in a HDF5 file correct key byte is 43

    # f = h5py.File('WS1.h5')
    #
    # dset = f['WS1Data']
    # inputs = np.array(dset[:, 0], dtype='int')
    # traces = dset[:, 1:5001]
    # len_traces = traces.shape[1]
    # f.close()

    if dosimMasking:
        sim_attack_masking(500)

    if doSNR:
        # This is currently configured for the AT89_SubBytes.h5 dataset
        ark = AddRoundKey(inputs, 119)
        sb = SubBytes[ark]
        hw_inputs = HW[inputs]
        hw_sb = HW[sb]
        hw_ark = HW[ark]
        labels =  hw_sb

        snr = SNR(9,labels, traces)
        fig, ax = plt.subplots()

        x = np.arange(0,len_traces,1)
        ax.set_xlabel('time')
        ax.set_ylabel('SNR')
        ax.plot(snr, color='blue')
        fig.savefig("SNR.png")

    if doCorr:
        # Call a correlation distinguisher
        print(f"[+] ")
        print(f"[+] Producing correlation distinguisher results ...")
        pcor_vals = PCor(inputs, traces)
        # find max value and return as key index
        key_ind = np.unravel_index(np.argmax(np.absolute(pcor_vals)), pcor_vals.shape)
        print(f"[+] Highest correlation occurs for key {key_ind[0]} at time index {key_ind[1]}.")
        # plot the Corr results
        fig, ax = plt.subplots()
        ax.set_xlabel('time')
        ax.set_ylabel('Correlation value')

        x = np.arange(len_traces)
        for i in range(256):
            ax.plot(x, pcor_vals[i, :], color='silver')
        ax.plot(x, pcor_vals[key_ind[0], :], color='black')
        ax.set_title('Correlation results: best key is {:d}'.format(key_ind[0]))
        fig.savefig("Corr.png")

    if doDOM:
        # Call a difference of means distinguisher
        print(f"[+] ")
        print(f"[+] Producing difference of means distinguisher results ...")
        dom_vals = DoM(inputs, traces)
        key_ind_dom = np.unravel_index(np.argmax(np.absolute(dom_vals)), dom_vals.shape)
        print(f"[+] Highest DoM value occurs for key {key_ind_dom[0]} at time index: {key_ind_dom[1]}.")
        # plot the DOM results
        fig, ax = plt.subplots()
        ax.set_xlabel('time')
        ax.set_ylabel('DoM value')

        x = np.arange(len_traces)
        for i in range(256):
            ax.plot(x, dom_vals[i, :], color='silver')
        ax.plot(x, dom_vals[key_ind_dom[0], :], color='black')
        ax.set_title('DoM results: best key is {:d}'.format(key_ind_dom[0]))
        fig.savefig("DoM.png")