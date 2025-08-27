import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# # Categories (x-axis labels)
loss_funcs = ['MSE', 'Constraint', 'Constraint w/ MSE', 'PBL']

# # PFNet Data
# means = [0.4804333333, 5.4619, 3.074266667]
# std_devs = [0.04925812962, 0.03566062254, 0.04025870506]

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

# CANOS Data
# means = [0.3464, 0.6430, 0.2129, 0.0804]
# std_devs = [0.0110, 0.0099, 0.0366, 0.0061]

# # Plot
# plt.figure(figsize=(6, 4))
# plt.errorbar(x, means, yerr=std_devs, fmt='o', capsize=5, color='black', ecolor='gray')

# # Set x-axis ticks to category labels
# plt.xticks(x, loss_funcs)
# plt.title('CANOS trained on various loss functions')
# plt.ylabel('Power Balance Loss')
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig('canos_pbl_on_4_loss_funcs.png')


import matplotlib.pyplot as plt
import numpy as np

loss_functions = ("MSE", "Constraint", "Constraint w/ MSE", "PBL")
# first tuple is means, second tuple is stddevs
pbl_means = {
    'CANOS': [(0.3464, 0.6430, 0.2129, 0.0804),(0.0110, 0.0099, 0.0366, 0.0061)],
    'PFNet': [(0.4804, 5.4619, 3.0743, 0.1035), (0.0493, 0.0357, 0.0403, 0.0080)]
}

mse_means = {
    'CANOS': [(0.14865, 575.32335, 0.1579, 25.4867), (0.00007, 0.7925,0.1202, 0.4001)],
    'PFNet': [(0.0039, 59.2362, 0.01625, 0.8293), (0.0016, 3.0288, 0.0064, 0.0057)]
}

x = np.arange(len(loss_functions))  # the label locations
width = 0.30  # the width of the bars
multiplier = 0

fig, (ax, ax2) = plt.subplots(2, 1, layout='constrained')
fig.set_size_inches(14, 6)
for attribute, measurement in pbl_means.items():
    means = measurement[0]
    yerr = measurement[1]

    offset = width * multiplier
    rects = ax.bar(x + offset, means, width, label=attribute, yerr=yerr)
     
    labels = [f"{x:.3f}" for x in means]
    ax.bar_label(rects, padding=3, labels=labels)
    multiplier += 1

multiplier = 0
for attribute, measurement in mse_means.items():
    means = measurement[0]
    yerr = measurement[1]

    offset = width * multiplier
    rects = ax2.bar(x + offset, means, width, label=attribute, yerr=yerr)
     
    labels = [f"{x:.3f}" for x in means]
    ax2.bar_label(rects, padding=3, labels=labels)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Validation PBL Mean Error')
ax.set_title('CANOS vs PFNet PBL on Various Loss Functions')
ax.set_xticks(x + width/2, loss_functions)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 6)
# ax.set_yscale('log')
# ax.set_ylim(1e-6, 1e1)
# ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
# ax.yaxis.set_major_formatter(mticker.LogFormatter())

ax2.set_ylabel('Validation MSE Error')
ax2.set_title('CANOS vs PFNet MSE trained on Various Loss Functions')
ax2.set_xticks(x + width/2, loss_functions)
ax2.legend(loc='upper left', ncols=3)

ax2.set_yscale('log')                 
ax2.set_ylim(1e-4, 1e4)              
ax2.yaxis.set_major_locator(mticker.LogLocator(base=10))
ax2.yaxis.set_major_formatter(mticker.LogFormatter())

# Resize font
for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
    text.set_fontsize(16)
ax.tick_params(axis='both', labelsize=16)
ax.legend(fontsize=16)

for text in (ax2.title, ax2.xaxis.label, ax2.yaxis.label):
    text.set_fontsize(16)
ax2.tick_params(axis='both', labelsize=16)
ax2.legend(fontsize=16)

plt.savefig('canos_vs_pfnet_mse_on_4_loss_funcs.png')