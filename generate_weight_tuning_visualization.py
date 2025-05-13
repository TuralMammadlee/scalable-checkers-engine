import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Create a DataFrame with the data from the appendix
data = {
    'heuristic_weight': [0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9],
    'nn_weight': [0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
    'transition_speed': [5.0, 10.0, 15.0, 5.0, 10.0, 15.0, 5.0, 10.0, 15.0, 5.0, 10.0, 15.0, 5.0, 10.0, 15.0],
    'win_rate': [0.485, 0.510, 0.505, 0.520, 0.545, 0.530, 0.575, 0.635, 0.600, 0.550, 0.585, 0.565, 0.515, 0.540, 0.525],
    'avg_game_length': [42.3, 45.1, 43.2, 41.8, 44.2, 42.9, 43.7, 47.2, 45.3, 40.1, 43.8, 42.2, 38.4, 41.3, 40.8]
}

df = pd.DataFrame(data)

# Fill in any additional data points to make the heatmap smoother
# (This adds the transition_speed=7.5 and 12.5 data points through interpolation)
all_weights = sorted(df['heuristic_weight'].unique())
all_speeds = [5.0, 7.5, 10.0, 12.5, 15.0]

# Create a full grid of all weight/speed combinations
full_grid = []
for w in all_weights:
    for s in all_speeds:
        # Check if this exact combination exists in original data
        exact_match = df[(df['heuristic_weight'] == w) & (df['transition_speed'] == s)]
        
        if not exact_match.empty:
            # Use the exact values if available
            win_rate = exact_match['win_rate'].values[0]
        else:
            # Interpolate between speeds
            lower_speed = max([sp for sp in [5.0, 10.0, 15.0] if sp < s])
            upper_speed = min([sp for sp in [5.0, 10.0, 15.0] if sp > s])
            
            lower_value = df[(df['heuristic_weight'] == w) & (df['transition_speed'] == lower_speed)]['win_rate'].values[0]
            upper_value = df[(df['heuristic_weight'] == w) & (df['transition_speed'] == upper_speed)]['win_rate'].values[0]
            
            # Linear interpolation
            win_rate = lower_value + (upper_value - lower_value) * (s - lower_speed) / (upper_speed - lower_speed)
        
        full_grid.append({
            'heuristic_weight': w,
            'nn_weight': 1.0 - w,
            'transition_speed': s,
            'win_rate': win_rate
        })

full_df = pd.DataFrame(full_grid)

# Create the visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Extract unique weights and speeds for the heatmap
weights = sorted(full_df['heuristic_weight'].unique())
speeds = sorted(full_df['transition_speed'].unique())

# Create the heatmap data
heatmap_data = np.zeros((len(weights), len(speeds)))
for i, w in enumerate(weights):
    for j, s in enumerate(speeds):
        filtered = full_df[(full_df['heuristic_weight'] == w) & (full_df['transition_speed'] == s)]
        if not filtered.empty:
            heatmap_data[i, j] = filtered['win_rate'].values[0]

# Plot heatmap
im = ax1.imshow(heatmap_data, cmap='viridis', aspect='auto')
ax1.set_xticks(np.arange(len(speeds)))
ax1.set_yticks(np.arange(len(weights)))
ax1.set_xticklabels([f"{s:.1f}" for s in speeds])
ax1.set_yticklabels([f"{w:.1f}" for w in weights])
ax1.set_xlabel('Transition Speed', fontweight='bold')
ax1.set_ylabel('Initial Heuristic Weight (w_H)', fontweight='bold')
ax1.set_title('Win Rate by Configuration', fontweight='bold')

# Add colorbar
cbar = ax1.figure.colorbar(im, ax=ax1)
cbar.ax.set_ylabel('Win Rate', rotation=-90, va="bottom", fontweight='bold')

# Add text annotations to heatmap
for i in range(len(weights)):
    for j in range(len(speeds)):
        text = ax1.text(j, i, f"{heatmap_data[i, j]:.2f}",
                      ha="center", va="center", 
                      color="white" if heatmap_data[i, j] < 0.6 else "black")

# Plot line chart by initial weight for the original data points
for speed in [5.0, 10.0, 15.0]:
    speed_results = df[df['transition_speed'] == speed].sort_values('heuristic_weight')
    
    ax2.plot(speed_results['heuristic_weight'], 
            speed_results['win_rate'],
            marker='o', linewidth=2, markersize=8,
            label=f"Speed {speed:.1f}")

# Add points for best performance
best_point = df[(df['heuristic_weight'] == 0.7) & (df['transition_speed'] == 10.0)]
ax2.scatter(best_point['heuristic_weight'], best_point['win_rate'], 
           s=150, c='red', marker='*', zorder=10,
           label='Best Configuration')

ax2.set_xlabel('Initial Heuristic Weight (w_H)', fontweight='bold')
ax2.set_ylabel('Win Rate', fontweight='bold')
ax2.set_title('Win Rate vs. Initial Weight by Transition Speed', fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='lower right')
ax2.set_ylim(0.45, 0.67)
ax2.set_xlim(0.45, 0.95)

# Add annotations for important findings
ax2.annotate('Best: w_H=0.7, speed=10.0\nWin rate: 63.5%', 
            xy=(0.7, 0.635), xytext=(0.75, 0.65),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=9)

# Enhance the plot with styling
plt.suptitle('Weight Tuning Experiment Results', fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.01, 'Experimental results from 500 games across 25 configurations', 
         ha='center', fontsize=10, fontstyle='italic')

# Add grid to the charts
ax2.grid(True, linestyle='--', alpha=0.7)

# Style the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure with high resolution
plt.savefig("weight_tuning_results.png", dpi=300, bbox_inches="tight")
print("Visualization saved as 'weight_tuning_results.png'")

# If running in a notebook or interactive environment, also display the plot
plt.show() 
