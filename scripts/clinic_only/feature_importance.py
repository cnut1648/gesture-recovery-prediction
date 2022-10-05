d = {
"ASA": 0.077225,
"Nerve Sparing":                         0.076636,
"ECE":                         0.075528,
"Radiation after surgery": 0.075329,
"Postop ADT": 0.077734,
"Age": 0.075985,
"BMI":0.075324,
"CCI": 0.075378,
"PSA":0.076456,
"Prostate volume (g)": 0.076170,
"Pre-op Gleason": 0.074284,
"Post-op Gleason ": 0.076537,
}
import numpy as np
import matplotlib.pyplot as plt
import torch
k = np.array(list(d.keys()))
# d = torch.tensor(list(d.values())).softmax(0).numpy()
d = torch.tensor(list(d.values())).numpy()

sorted_idx = d.argsort()
plt.figure()
plt.barh(k[sorted_idx], d[sorted_idx])
plt.savefig("ff.png")
