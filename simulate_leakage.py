import numpy as np
import scipy.stats as stats
import dpa

s = 0.1
mu = 0
sigma = s

# Resulting power trace is (256, 6). Each input (key byte for S box) generates 6 time points
# The target for second task is time point 2
def leak_values(S, m, mprime):
    Sm = np.zeros((S.shape[0]))
    power_trace = np.zeros((256, 6))
    for i in range(256):
        power_trace[i, 0] = dpa.HW[i] + stats.norm.rvs(mu, sigma)
        power_trace[i, 1] = dpa.HW[m] + stats.norm.rvs(mu, sigma)
        power_trace[i, 2] = dpa.HW[mprime] + stats.norm.rvs(mu, sigma)
        power_trace[i, 3] = dpa.HW[i ^ m] + stats.norm.rvs(mu, sigma)
        power_trace[i, 4] = dpa.HW[S[i ^ m]] + stats.norm.rvs(mu, sigma)
        power_trace[i, 5] = dpa.HW[S[i ^ m] ^ mprime] + stats.norm.rvs(mu, sigma)
        Sm[i] = S[i ^ m] ^ mprime

    return Sm, power_trace

m = 42
mprime = m
Sm_1, power_trace_m = leak_values(dpa.SubBytes, m, mprime) #power trace is (256, 6) 1 input is a key byte calculation in the table

# attempt DPA using candidate m values where m == m', since m is just a byte
# We need to predict, for each input, what the target is. Our prediction is just the HW of m

def attack_m_equals_mprime(power_trace):
    pred_leak = np.zeros((256,256))
    inputs = np.arange(256)
    # We need to iterate over all 256 possible values of m
    # And the predicted leakage needs to do 256 bytes from the table
    for m_candidate in range(256):
            pred_leak[m_candidate, :] = dpa.HW[dpa.SubBytes[inputs ^ m_candidate]]

    pcor_correlation = dpa.compute_correlation(pred_leak.T, power_trace) # (256, 6)
    candidate_ranking = np.argsort(np.max(np.absolute(pcor_correlation), axis=1))[::-1]
    for i in range(5):
        print(f"Candidate for m ranked {i+1} is {candidate_ranking[i]}")

print(f"[+] Setting m and m' to {m}")
print(f"[+] Simulating power leakage and recovering m")
attack_m_equals_mprime(power_trace_m)
# find most likely m from correlation

m = 9
mprime = 80
Sm_2, power_trace_mprime = leak_values(dpa.SubBytes, m, mprime)

# attempt DPA using candidate m and m' values where m != m'
def attack_m_nequals_mprime(power_trace):
    inputs = np.arange(256)

    # predict m
    pred_leak = np.zeros((256, 256, 256))
    pcor_correlation = np.zeros((256, 256, 6))
    combined_correlation = np.zeros((256, 256))
    for m_prime_candidate in range(256):
        for m_candidate in range(256):
            pred_leak[m_prime_candidate, m_candidate, :] = dpa.HW[dpa.SubBytes[inputs ^ m_candidate] ^ m_prime_candidate]
        pcor_correlation[m_prime_candidate, :, :] = dpa.compute_correlation(pred_leak[m_prime_candidate, :, :], power_trace)
        combined_correlation[m_prime_candidate, :] = np.mean(pcor_correlation[m_prime_candidate, :, :], axis=1)

    # Compute the maximum value at each (x, y) coordinate across the 6 channels
    max_correlations = np.max(pcor_correlation, axis=-1)  # shape (256, 256)
    flattened_indices = np.argsort(max_correlations.flatten())[::-1]
    pcor_rankings = np.array(np.unravel_index(flattened_indices, max_correlations.shape)).T

    most_likely_m_prime, most_likely_m = pcor_rankings[0, :]
    max_indices = np.unravel_index(combined_correlation.argmax(), combined_correlation.shape)
    print(f"Top ranked pair using max correlation of m and mprime is: {most_likely_m} and {most_likely_m_prime} whereas top ranked pair using aggregate correlation is: {max_indices[1]} and {max_indices[0]}")
    # predict mprime

print(f"[+] Setting m to {m} and m' to {mprime}")
print(f"[+] Simulating power leakage 20 times and calculating top candidates")
for i in range(20):
    Sm_2, power_trace_mprime = leak_values(dpa.SubBytes, m, mprime)
    attack_m_nequals_mprime(power_trace_mprime)


# Calculate SNR
