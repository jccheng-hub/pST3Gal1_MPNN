#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_ori = pd.read_csv('rmsf_data_edit.csv',index_col='Residue Number')
df = df_ori.T

scaler = StandardScaler(with_std=False)
centered_data = scaler.fit_transform(df)

pca = PCA(n_components=3)
principal_components = pca.fit_transform(centered_data)

scores = pd.DataFrame(data=principal_components,
                      columns=['PC1','PC2','PC3'],
                      index=df.index)

loadings = pca.components_
variance_explained = pca.explained_variance_ratio_

# Adjust colors
default_color = 'lightblue'
color_map = {
    'Wild Type': 'orange',
    'Variant 3': 'blue',
    'Variant 9': 'blue'
}
point_colors = [color_map.get(var, default_color) for var in scores.index]

# --- 2D plot for PC1 vs PC2 --- # {{{
plt.figure(figsize=(8,6))
plt.scatter(scores['PC1'], scores['PC2'], alpha=0.7, c=point_colors, s=100)
fontSize=14

# Annotate points with variant names
for variant in scores.index:
    if variant == 'Variant 1':
        x_nudge, y_nudge = -68, -20
    elif variant == 'Variant 2':
        x_nudge, y_nudge = 6, 3
    elif variant == 'Variant 6':
        x_nudge, y_nudge = -68, -20
    elif variant == 'Variant 7':
        x_nudge, y_nudge = -5, -20
    elif variant == 'Variant 9':
        x_nudge, y_nudge = 6, 3
    elif variant == 'Variant 10':
        x_nudge, y_nudge = 6, 3
    else:
        x_nudge, y_nudge = 6, -20
    plt.annotate(
        variant,
        (scores.loc[variant, 'PC1'], scores.loc[variant, 'PC2']),
        xytext=(x_nudge, y_nudge),
        textcoords='offset points',
        ha='left', va='bottom',
        fontsize=fontSize
    )

plt.xticks(fontsize=fontSize)
plt.yticks(fontsize=fontSize)
plt.xlabel(f'PC1 ({variance_explained[0]*100:.1f}% Variance)', fontsize=fontSize)
plt.ylabel(f'PC2 ({variance_explained[1]*100:.1f}% Variance)', fontsize=fontSize)
plt.axhline(0, color='black', linewidth=0.5, ls='--', alpha=0.2)
plt.axvline(0, color='black', linewidth=0.5, ls='--', alpha=0.2)
plt.tight_layout()
plt.savefig('pca.png', dpi=300)
# plt.show()
# }}}

# # --- 3D plot for PC1 vs PC2 vs PC3 --- # {{{
# fig = go.Figure()
# scores['Color'] = [color_map.get(var, default_color) for var in scores.index]
# 
# # Add the scatter points
# fig.add_trace(go.Scatter3d(
#     x=scores['PC1'],
#     y=scores['PC2'],
#     z=scores['PC3'],
#     mode='markers+text',
#     text=scores.index,        # Attaches the variant name to the hover/label
#     textposition='top center',
#     marker=dict(
#         size=6,
#         color=scores['Color'],
#         opacity=0.8,
#         line=dict(width=0.5, color='black') # Slight border around dots
#     )
# ))
# 
# # Update the layout with proper axis labels showing variance
# fig.update_layout(
#     title='RMSF PCA: 3D Variant Clustering',
#     scene=dict(
#         xaxis_title=f'PC1 ({variance_explained[0]*100:.1f}%)',
#         yaxis_title=f'PC2 ({variance_explained[1]*100:.1f}%)',
#         zaxis_title=f'PC3 ({variance_explained[2]*100:.1f}%)',
#     ),
#     width=900,
#     height=700
# )
# 
# fig.write_html('pca_3d_interactive.html')
# print("Saved 3D plot to pca_3d_interactive.html")
# # }}}

# --- Loadings plot --- # {{{
fixedres = [
    104, 105, 106, 107, 108, 109, 110, 111, 147, 148, 149, 150, 151, 152, 153, 
    154, 170, 171, 172, 173, 174, 175, 176, 191, 192, 193, 194, 195, 196, 197, 
    198, 199, 209, 210, 211, 212, 213, 215, 216, 231, 232, 233, 234, 235, 236, 
    255, 263, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 289, 290, 
    291, 292, 293, 294, 295, 296, 298, 299, 300, 301, 302, 303, 304, 305, 307, 
    308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 
    323, 324, 325, 326, 328, 339, 340, 341, 343
]
span_ranges = []
startnum = fixedres[0]
prevnum = fixedres[0]
for current_num in fixedres[1:]:
    if current_num - prevnum > 1:
        span_ranges.append((startnum, prevnum))
        startnum = current_num   
    prevnum = current_num
span_ranges.append((startnum, prevnum))

pc1_loadings = loadings[0]
pc2_loadings = loadings[1]
# pc3_loadings = loadings[2]
residues = df.columns

fig, ax = plt.subplots(figsize=(9,6))
# plt.figure(figsize=(10, 5))
ax.plot(residues, pc1_loadings, label='PC1 influence', color='red', alpha=0.5)
ax.plot(residues, pc2_loadings, label='PC2 influence', color='blue', alpha=0.5)
# plt.plot(residues, pc3_loadings, label='PC3 influence', color='purple', alpha=0.5)

# Highlight
for start, end in span_ranges:
    ax.axvspan(start, end, color='cyan', alpha=0.1)

ax.axhline(0, color='black', linewidth=1, ls='--')

def add_range_label(x_start, x_end, tick_length, y_brace):
    ax.plot([x_start, x_start, x_end, x_end], 
            [y_brace - tick_length, y_brace, y_brace, y_brace - tick_length], 
            color='black', lw=1)
    if tick_length < 0:
        ax.text((x_start + x_end) / 2, y_brace - 0.005, f'{x_start}-{x_end}', 
                ha='center', va='top', fontsize=fontSize-2)
    else:
        ax.text((x_start + x_end) / 2, y_brace + 0.005, f'{x_start}-{x_end}', 
                ha='center', va='bottom', fontsize=fontSize-2)

add_range_label(298, 311, 0.01, 0.38)
add_range_label(148, 155, 0.01, 0.075)
add_range_label(173, 181, 0.01, 0.055)
add_range_label(227, 238, -0.01, -0.135)

ax.set_xlabel('Residue number', fontsize=fontSize)
ax.set_ylabel('Loading weight', fontsize=fontSize)
ax.set_xlim(60, 343)
ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.01, 1), fontsize=fontSize)

plt.xticks(fontsize=fontSize)
plt.yticks(fontsize=fontSize)
plt.tight_layout()
plt.savefig('loadings_plot.png', dpi=300)

#indices = np.where(loadings[1] > 0.03)
#indices = np.where(loadings[1] < -0.08)
#print(residues[indices])
# }}}
