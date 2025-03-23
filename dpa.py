import h5py
import numpy as np
import matplotlib.pyplot as plt
import threading

from readchar import key
from scipy import stats

# The AES SBox that we will use to generate our labels
SubBytes = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])
HW = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
               3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
               3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
               2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
               3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
               5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
               2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
               4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
               3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
               4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
               5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
               5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8])


def AddRoundKey(plain, key):
    return np.bitwise_xor(plain, key)


def getLSB(value):
    return value & 1

def DOM_Vectorized(inputs, traces):
    trace_len = traces.shape[1]
    # iterate over all 256 values that a key byte can take
    dom_rankings = np.zeros((16, 5), dtype=int)
    input_len = inputs.shape[0]

    for key_byte in range(inputs.shape[1]):
        dom_dist = np.zeros((256, trace_len))
        input_key_bytes = inputs[:, key_byte]

        precomputed_LSB = np.zeros((256, input_len))
        for key_candidate in range(256):
            precomputed_LSB[key_candidate, :] = getLSB(SubBytes[AddRoundKey(input_key_bytes, key_candidate)])

        correlations = np.zeros((256, trace_len))

        mask0 = (precomputed_LSB == 0).astype(float)
        mask1 = 1.0 - mask0


        b0_sum = mask0.dot(traces)
        b0_count = mask0.sum(axis=1, keepdims=True)
        b0_count[b0_count == 0] = 1
        b0_mean = b0_sum / b0_count

        b1_sum = mask1.dot(traces)
        b1_count = mask1.sum(axis=1, keepdims=True)
        b1_count[b1_count == 0] = 1
        b1_mean = b1_sum / b1_count


        b0_second = mask0.dot(traces ** 2) / b0_count
        b0_var = b0_second - b0_mean ** 2

        b1_second = mask1.dot(traces ** 2) / b1_count
        b1_var = b1_second - b1_mean ** 2

        dom_dist = (b0_mean - b1_mean) / np.sqrt(b0_var + b1_var)

        # for key_candidate in range(256):
        #     precomputed_LSB[key_candidate, :] = getLSB(SubBytes[AddRoundKey(input_key_bytes, key_candidate)])
        #
        #     # calculate the target
        #     mask0 = precomputed_LSB[key_candidate, :] == 0
        #     mask1 = ~mask0
        #     # apply a single bit leakage model and categorise traces
        #     bucket0 = traces[mask0, :]
        #     bucket1 = traces[mask1, :]
        #
        #     b0_mean = np.mean(bucket0, axis=0)
        #     b1_mean = np.mean(bucket1, axis=0)
        #
        #     b0_std = np.var(bucket0, axis=0)
        #     b1_std = np.var(bucket1, axis=0)
        #     # calculate distinguisher value
        #     dom_dist[key_candidate, :] = (b0_mean - b1_mean) / (np.sqrt(b0_std + b1_std))
        #
        dom_rankings[key_byte, :] = np.argsort(np.max(np.absolute(dom_dist), axis=1))[:5]
        print(f"Done key byte {key_byte}")

    return dom_rankings


def DOM_HW(inputs, traces):
    trace_len = traces.shape[1]
    # iterate over all 256 values that a key byte can take
    dom_rankings = np.zeros((16, 5), dtype=int)
    input_len = inputs.shape[0]

    for key_byte in range(inputs.shape[1]):
        dom_dist = np.zeros((256, trace_len))
        input_key_bytes = inputs[:, key_byte]
        precomputed_LSB = np.zeros((256, input_len))


        correlations = np.zeros((256, trace_len))
        for key_candidate in range(256):
            precomputed_LSB[key_candidate, :] = getLSB(SubBytes[AddRoundKey(input_key_bytes, key_candidate)])

            # calculate the target
            mask0 = precomputed_LSB[key_candidate, :] == 0
            mask1 = ~mask0
            # apply a single bit leakage model and categorise traces
            bucket0 = traces[mask0, :]
            bucket1 = traces[mask1, :]

            b0_mean = np.mean(bucket0, axis=0)
            b1_mean = np.mean(bucket1, axis=0)


            b0_std = np.var(bucket0, axis=0)
            b1_std = np.var(bucket1, axis=0)
            # calculate distinguisher value
            dom_dist[key_candidate, :] = (b0_mean - b1_mean) / (np.sqrt(b0_std + b1_std))


        dom_rankings[key_byte, :] = np.argsort(np.max(np.absolute(dom_dist), axis=1))[:5]
        print(f"Done key byte {key_byte}")


    return dom_rankings

def DoM(inputs, traces):
    trace_len = traces.shape[1]
    # iterate over all 256 values that a key byte can take
    dom_dist = np.zeros((256, trace_len))




    for key_guess in range(256):
        # calculate the target
        ark = AddRoundKey(inputs, key_guess)
        sb = SubBytes[ark]
        # apply a single bit leakage model and categorise traces
        bucket0 = traces[getLSB(sb) == 0, :]
        bucket1 = traces[getLSB(sb) == 1, :]
        # calculate distinguisher value
        dom_dist[key_guess, :] = (np.mean(bucket0, axis=0) - np.mean(bucket1, axis=0))/(np.sqrt(np.var(bucket0,axis=0)+np.var(bucket1, axis=0)))

    return dom_dist

# Calculate PMCC
def compute_correlation(hw_matrix, traces):
    # Calculate mean of input HW for each key candidate
    # And mean of all input leakage values for each time point
    HW_mean = np.mean(hw_matrix, axis=0)
    T_mean = np.mean(traces, axis=0)

    # Compute covariance
    # Cov(X, Y) = E(X - E(X)) * E(Y - E(Y))
    # So calculate differences between each predicted HW value for an input and mean for each key candidate
    # (each row corresponds to an input from the power trace and consists of predicted HW values from key candidates)
    # and difference between each leakage value and the mean value across all inputs at that time point
    HW_diff = hw_matrix - HW_mean
    T_diff = traces - T_mean
    # Calculate SUM (X - E(X)) * (Y - E(Y))
    # 1/n is cancelled out from the denominator
    covariance = np.dot(HW_diff.T, T_diff)

    # Compute the standard deviations (denominators)
    # Calculate std of predicted HW across each input for each key candidate
    # Calculate std of leakage values across each input for each time point
    HW_std = np.sqrt(np.sum(HW_diff ** 2, axis=0))  # Shape (256,)
    T_std = np.sqrt(np.sum(T_diff ** 2, axis=0))  # Shape (100000,)

    # Final correlation matrix
    # When we multiply on the denominator, this creates a matrix equivalent in shape
    # to the covariance matrix.
    # Intuitively, when we did np.dot(HW_diff.T, T_Diff), for a specific input from the power trace
    # we multiplied its mean centered HW (calculated from the corresponding key candidate) with
    # the mean centered leakage value for the same input at that time point. Then when we sum,
    # we summed it over all inputs. Thus entry covariance[0,0] gives us covariance between the first key candidate byte (0x0)
    # and the first time point
    # Similarly covariance[0, 1] gives cov for first key candidate and 2nd time point
    # And similarly covariance[1,0] gives cov for second key candidate and first time point.
    # When dividing, we essentially normalise them all to between [-1, 1]
    # Thus we find the rows (key candidates) with the highest correlation peak (that index aka time point is where it is
    # has the highest correlation with our predicted leakages)
    correlation_matrix = covariance / (HW_std[:, np.newaxis] * T_std)
    return correlation_matrix

def Pcor_Vectorized(inputs, traces, high_snr_indices):

    trace_len = int(traces.shape[1] * 0.1)
    input_len = inputs.shape[0]
    pcor_rankings = np.zeros((16, 5), dtype=int)

    for key_byte in range(inputs.shape[1]):
        input_key_bytes = inputs[:, key_byte]
        precomputed_HW = np.zeros((256, input_len))
        correlations = np.zeros((256, trace_len))
        for key_candidate in range(256):
            precomputed_HW[key_candidate, :] = HW[SubBytes[AddRoundKey(input_key_bytes, key_candidate)]]

        traces_to_consider = traces[:, high_snr_indices[key_byte, :]]
        correlations = compute_correlation(precomputed_HW.T, traces_to_consider)
        pcor_rankings[key_byte, :] = np.argsort(np.max(np.absolute(correlations), axis=1))[:5]

        highest_correlation_key_byte = pcor_rankings[key_byte, 0]
        print(f"Done byte {key_byte}")

    return pcor_rankings

def Pcor_HW(inputs, traces):
    trace_len = traces.shape[1]
    input_len = inputs.shape[0]
    pcor_rankings = np.zeros((16, 5), dtype=int)

    for key_byte in range(inputs.shape[1]):
        input_key_bytes = inputs[:, key_byte]
        precomputed_HW = np.zeros((256, input_len))
        correlations = np.zeros((256, trace_len))
        for key_candidate in range(256):
            precomputed_HW[key_candidate, :] = HW[SubBytes[AddRoundKey(input_key_bytes, key_candidate)]]

        correlations = compute_correlation(precomputed_HW.T, traces)
        pcor_rankings[key_byte, :] = np.argsort(np.max(np.absolute(correlations), axis=1))[:5]

        highest_correlation_key_byte = pcor_rankings[key_byte, 0]
        print(f"Done byte {key_byte}")

    return pcor_rankings

def PCor(inputs, traces):
    # iterate over all the 256 values that a key byte can take
    trace_len = traces.shape[1]
    pcor_dist = np.zeros((256, trace_len))
    for key_candidate in range(256):
        # calculate the target
        ark = AddRoundKey(inputs, key_candidate)
        sb = SubBytes[ark]
        # apply a Hamming weight leakage model
        pred_leak = HW[sb]
        # pred_leak = getLSB(sb)
        #pred_leak = HW[ark]
        chunksize = 25000
        for chunk in range(0, trace_len, chunksize):
            cor = np.corrcoef(pred_leak.T, traces[:, chunk:chunk+chunksize], rowvar=False)
            pcor_dist[key_candidate, chunk:chunk+chunksize] = cor[0, 1:]
        print(f"Done key candidate {key_candidate}")
    return pcor_dist

def mean_center_traces(traces):
    return traces - np.mean(traces, axis=0)



# num_labels = 9, labels = (256,200), traces = (200, 100000)
def SNR_Vectorized(num_labels, labels, traces):

    # Mean center traces
    mean_centered_traces = mean_center_traces(traces)
    num_candidates, num_inputs = labels.shape
    num_traces = traces.shape[1]

    candidate_exp = np.zeros((num_candidates, num_labels, num_traces)) # 256, 9, 100000
    candidate_var = np.zeros((num_candidates, num_labels, num_traces)) # 256, 9, 100000

    for h in range(num_labels):
        mask = (labels == h).astype(int) # Mask of labels (256, 200) where HW is h so is (256, 200)
        counts = mask.sum(axis=1) # Gets count how many HW is h per key candidate so is (256,)
        counts[counts == 0] = 1 # Sets where count == 0 to 1 to prevent dividing by zero

        # Gets candidate_exp[:, h, :] is shape (256, 1, 200) so sets, for each candidate and input,
        # mask.dot(traces) does this:
        # For each candidate key, its row in mask represents which inputs have labels == h
        # Then, for each input, multiplies by its respective input's leakage value for each timepoint. So if its == 1
        # then it uses the input's leakage value, otherwise doesnt. Then sums it up and divides by number
        # of 1s (number of inputs that have label == h). So this gets the mean of the TIME POINTS where
        # its predicted leakage is h
        candidate_exp[:, h, :] = mask.dot(mean_centered_traces) / counts[:, None]

        # Does above but calculates leakage value ^ 2 for variance calculation
        candidate_second_moment = mask.dot(mean_centered_traces ** 2) / counts[:, None]

        # Calculates variances since Var(X) = E(X^2) - E(X)^2
        candidate_var[:, h, :] = candidate_second_moment - candidate_exp[:, h, :] ** 2

    signal = np.var(candidate_exp, axis=1)
    noise = np.mean(candidate_var, axis=1)
    noise[noise == 0] = np.finfo(float).eps
    snr = signal / noise
    return snr

# num_labels = 9, labels = (200,), traces = (200, 100000)
def SNR_HW(num_labels, labels, traces):
    num_points = traces.shape[1]
    exp = np.zeros((num_labels, num_points)) # 9,100000
    std = np.zeros((num_labels, num_points)) # 9,100000
    for h in range(num_labels):
        # finds HW predictions (labels) which == h
        ind = np.nonzero(labels == h)
        # Gets the inputs where HW == h from traces
        # and calculates the mean of the TIME POINTS
        exp[h, :] = np.mean(traces[ind, :], axis=1)
        # Same thing here but calculates VARIANCE
        std[h, :] = np.var(traces[ind, :], axis=1)

    # sample variance of signal, which is exp_hw
    # For each predicted HW from input, calculates the variance (signal)
    # Does this since labels is [2, 4, 1, 6, 7, ...] HW values so
    # exp[labels] returns array of (200, 100000) where each entry corresponds to
    # mean of timepoints for inputs where HW = labels[i]
    # Then we take the variance for each input
    # so signal ends up as (100000,)
    signal = np.var(exp[labels], axis=0)
    # Same thing for noise (mean)
    noise = np.mean(std[labels], axis=0)

    snr = signal / noise
    return snr

def SNR(num_labels, labels, traces):
    num_points = traces.shape[1]
    exp= np.zeros((num_labels, num_points))
    std= np.zeros((num_labels, num_points))
    for h in range(num_labels):
        ind = np.nonzero(labels == h)
        exp[h, :] = np.mean(traces[ind, :], axis=1)
        std[h, :] = np.var(traces[ind, :], axis =1)

    # sample variance of signal, which is exp_hw
    signal = np.var(exp, axis=0)
    noise = np.mean(std, axis=0)
    snr = signal/noise
    return snr

def sim_attack_masking(num_inputs):
    # create a random sequence of inputs, as well as a random key
    # we use a generator that draws from a uniform distribution
    # hint use a fixed seed if you want the same inputs to be produced over multiple runs
    rng = np.random.default_rng()
    inputs = rng.integers(low=0, high=256, size = (num_inputs,1))
    skey = rng.integers(low=0, high=256, size=1)
    print(f"[+] Secret key is supposed to be {skey}")
    ark = AddRoundKey(inputs, skey)
    sb = SubBytes[ark]

    # let's simulate the secret sharing of an intermediate value
    sb_share= rng.integers(low=0, high=255, size=(num_inputs,1))
    sb_share = np.concatenate((sb_share, np.bitwise_xor(sb_share,sb)),axis=1)

    # let's simulate HW leakage of the two shares
    traces = HW[sb_share] + rng.normal(0,1, size=(num_inputs,2))
    # add a third point with raw noise
    traces = np.concatenate((traces, rng.normal(1,1,size=(num_inputs,1))), axis=1)

    # traces need to be pre-processed, so that we create joint leakage between the shares
    # because there are just three points I do this by hand
    # in the pre-processed traces, the first trace point contains the joint leakage
    # between the shares, whereas the other two points are joint leakage of a share with some
    # independent noise
    traces = traces - np.mean(traces, axis=0)
    prod_traces =  np.zeros((num_inputs,3))
    prod_traces[:,0] = traces[:,0]*traces[:,1]
    prod_traces[:,1] = traces[:,0]*traces[:,2]
    prod_traces[:,2] = traces[:,1]*traces[:,2]

    # now we use the PCor function on the generated inputs and traces
    # make sure that inside PCor() you predict the same intermediate value and use a HW model

    inp = np.reshape(inputs, (1,-1))
    pcor_vals = PCor(inp, prod_traces)
    key_ind = np.unravel_index(np.argmax(np.absolute(pcor_vals)), pcor_vals.shape)
    print(f"[+] Highest correlation occurs for key {key_ind[0]} at time index {key_ind[1]}.")
    if skey==key_ind[0]:
        print(f"Yipee!")