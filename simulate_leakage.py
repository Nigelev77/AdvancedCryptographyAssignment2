import numpy as np
import scipy.stats as stats
import dpa

s = 0.1
mu = 0
sigma = s


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
Sm_1, power_trace_m = leak_values(dpa.SubBytes, m, mprime)

# attempt DPA using candidate m values where m == m', since m is just a byte

pred_leaks = np.zeros((256, 256, 6))
pcor_dist = np.zeros((256, 256 * 6))
for key_byte in range(256):
    pred_leak = leak_values(dpa.SubBytes, m, mprime)
    pcor_dist[key_byte] = np.corrcoef(pred_leak.T, power_trace_m, rowvar=False)[0, 1:]


# find most likely m from correlation

key_ind_m = np.unravel_index(np.argmax(np.absolute(pcor_dist)), pcor_dist.shape)
print(f"[+] Highest correlation occurs for mask byte {key_ind_m[0]} whereas the actual mask is {m}")


# attempt DPA using candidate m and m' values where m != m'

m = 21
mprime = 71
Sm_2, power_trace_mprime = leak_values(dpa.SubBytes, m, mprime)
