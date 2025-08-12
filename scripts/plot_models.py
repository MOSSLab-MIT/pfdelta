import matplotlib.pyplot as plt
import numpy as np

# # Categories (x-axis labels)
loss_funcs = ['MSE', 'Constraint', 'Constraint w/ MSE']

# # Data
# means = [0.4804333333, 2.309433333, 3.071833333]
# std_devs = [0.04925812962, 0.5050024191, 0.01778351296]

# # Map categories to numeric positions
x = np.arange(len(loss_funcs))

# # Plot
# plt.figure(figsize=(6, 4))
# plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, color='black', ecolor='gray')

# # Set x-axis ticks to category labels
# plt.xticks(x, loss_funcs)
# plt.title('PFNet trained on various loss functions')
# plt.ylabel('Power Balance Loss')
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig('pfnet_pbl_on_3_loss_funcs.png')

# Data
means = [0.3464, 0.6429666667, 0.2129]
std_devs = [0.0109672239, 0.009929921114, 0.03656446362]

# Plot
plt.figure(figsize=(6, 4))
plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, color='black', ecolor='gray')

# Set x-axis ticks to category labels
plt.xticks(x, loss_funcs)
plt.title('CANOS trained on various loss functions')
plt.ylabel('Power Balance Loss')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('canos_pbl_on_3_loss_funcs.png')