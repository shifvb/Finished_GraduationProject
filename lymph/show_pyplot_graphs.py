import matplotlib.pyplot as plt
import numpy as np

plt.plot([0, 1, 2], [0.881, 0.908, 0.794], 'r--', [0, 1, 2], [0.889, 0.962, 0.845])

plt.ylabel("Accuracy")
ax = plt.gca()
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Liver", "Lymphoma", "Background"])
plt.show()
