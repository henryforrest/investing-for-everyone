import matplotlib.pyplot as plt
import numpy as np

# Generate some data that resembles a stock chart
np.random.seed(5) #randomised seed until i found a graph i liked 
x = np.arange(0, 100)
y = np.cumsum(np.random.randn(100))

# Create the plot
plt.figure(figsize=(10, 4))
plt.plot(x, y, color='blue', linewidth=2)

# Customize the plot to make it look like a stock chart
plt.fill_between(x, y, color='blue', alpha=0.1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks([])
plt.yticks([])
plt.box(False)

# Save the plot as an image
plt.savefig('graph1.png', transparent=True, bbox_inches='tight', pad_inches=0)

plt.show()