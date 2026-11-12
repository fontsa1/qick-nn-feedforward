"""Class SNR of the replay channel, with its definition: rms separation of the class means over rms noise around them, per complex sample,
19,400 rows. Source: test traces through the domain map. Replayed: v1.44 NN_axi session. Lags come from paper_figures/fig4/fig4_data.py.
Writes campaign_v144/final/analysis/channel_v144.json."""
import json
import numpy as np
A = "campaign_v144/final/analysis/"
y = np.load("data/test20k_real_y.npy").astype(int); keep = np.ones(20000, bool); keep[1000:1600] = False; y = y[keep]
c = lambda X: X[:, 0::2] + 1j * X[:, 1::2]
def snr(z):
    a, b = z[y == 1].mean(0), z[y == 0].mean(0)
    n = np.concatenate([z[y == 1] - a, z[y == 0] - b])
    return float(np.sqrt((abs(a - b) ** 2).mean()) / np.sqrt((abs(n) ** 2).mean()))
k2, c2 = complex(np.load("calibration/k2_loopback.npy")), complex(np.load("calibration/c2_loopback.npy"))
out = dict(snr_source=snr((k2 * c(np.load("data/test20k_real_X.npy", mmap_mode="r")[:20000].astype(np.float64)) + c2)[keep]))
for b in "AB":
    out["snr_" + b] = snr(c(np.load(A + f"replay20k_{b}.npy").astype(np.float64))[keep])
out.update(json.load(open(A + "fig2_lags_v144.json")))
json.dump(out, open(A + "channel_v144.json", "w"), indent=1); print(out)
