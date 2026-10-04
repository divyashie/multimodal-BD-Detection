"""
Visualize Ablation Study Results - No Data Leakage
Creates comprehensive plots comparing modality performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Results from ablation study
results = {
    'Dataset': ['unimodal_T', 'unimodal_A', 'unimodal_V', 'bimodal_TA', 'bimodal_TV', 'bimodal_AV', 'trimodal_TAV'],
    'Description': ['Text\nOnly', 'Audio\nOnly', 'Video\nOnly', 'Text +\nAudio', 'Text +\nVideo', 'Audio +\nVideo', 'Text + Audio\n+ Video'],
    'Short_Name': ['T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'],
    'Type': ['Unimodal', 'Unimodal', 'Unimodal', 'Bimodal', 'Bimodal', 'Bimodal', 'Trimodal'],
    'Test_Accuracy': [88.13, 72.91, 72.91, 72.91, 73.12, 72.91, 73.32],
    'Test_F1': [87.04, 72.62, 72.64, 72.60, 72.83, 72.61, 73.05],
    'F1_Depression': [82.96, 60.87, 62.07, 60.87, 60.87, 60.87, 60.87],
    'F1_Mania': [93.35, 75.19, 75.19, 75.38, 75.33, 75.29, 75.38],
    'F1_Euthymia': [54.92, 73.47, 73.10, 73.16, 73.84, 73.31, 74.35],
    'Train_Samples': [13957, 2289, 2289, 2289, 2289, 2289, 2289],
    'Test_Samples': [2991, 491, 491, 491, 491, 491, 491]
}

df = pd.DataFrame(results)

# Create output directory
output_dir = Path('ablation_results_no_leakage/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("CREATING VISUALIZATIONS FOR ABLATION STUDY")
print("="*80)
print()

# ============================================================================
# PLOT 1: Overall Performance Comparison (Accuracy & F1)
# ============================================================================
print("Creating Plot 1: Overall Performance Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Color palette
colors = ['#2ecc71' if x == 'Unimodal' else '#3498db' if x == 'Bimodal' else '#e74c3c' for x in df['Type']]

# Plot 1a: Test Accuracy
x_pos = np.arange(len(df))
bars1 = ax1.bar(x_pos, df['Test_Accuracy'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Modality Combination', fontsize=13, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Test Accuracy by Modality Combination', fontsize=15, fontweight='bold', pad=20)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df['Description'], fontsize=10)
ax1.set_ylim([0, 100])
ax1.axhline(y=48.8, color='red', linestyle='--', linewidth=2, label='Majority Baseline (48.8%)', alpha=0.7)
ax1.axhline(y=df[df['Dataset'] == 'unimodal_T']['Test_Accuracy'].values[0],
            color='green', linestyle='--', linewidth=2, label='Text-Only (88.1%)', alpha=0.7)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, df['Test_Accuracy'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 1b: Test F1 Score
bars2 = ax2.bar(x_pos, df['Test_F1'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Modality Combination', fontsize=13, fontweight='bold')
ax2.set_ylabel('Test F1 Score (%)', fontsize=13, fontweight='bold')
ax2.set_title('Test F1 Score by Modality Combination', fontsize=15, fontweight='bold', pad=20)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df['Description'], fontsize=10)
ax2.set_ylim([0, 100])
ax2.axhline(y=df[df['Dataset'] == 'unimodal_T']['Test_F1'].values[0],
            color='green', linestyle='--', linewidth=2, label='Text-Only (87.0%)', alpha=0.7)
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, df['Test_F1'])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', alpha=0.8, edgecolor='black', label='Unimodal'),
    Patch(facecolor='#3498db', alpha=0.8, edgecolor='black', label='Bimodal'),
    Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='Trimodal')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'overall_performance.png'}")
plt.close()

# ============================================================================
# PLOT 2: Per-Class F1 Scores (Grouped Bar Chart)
# ============================================================================
print("Creating Plot 2: Per-Class F1 Scores...")

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(df))
width = 0.25

bars1 = ax.bar(x - width, df['F1_Depression'], width, label='Depression',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x, df['F1_Mania'], width, label='Mania',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
bars3 = ax.bar(x + width, df['F1_Euthymia'], width, label='Euthymia',
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)

ax.set_xlabel('Modality Combination', fontsize=13, fontweight='bold')
ax.set_ylabel('F1 Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Per-Class F1 Performance by Modality', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df['Description'], fontsize=10)
ax.set_ylim([0, 100])
ax.legend(fontsize=12, loc='lower right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'per_class_f1.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'per_class_f1.png'}")
plt.close()

# ============================================================================
# PLOT 3: Modality Type Comparison (Box Plot Style)
# ============================================================================
print("Creating Plot 3: Modality Type Comparison...")

fig, ax = plt.subplots(figsize=(10, 8))

# Group by modality type
unimodal = df[df['Type'] == 'Unimodal']['Test_F1'].values
bimodal = df[df['Type'] == 'Bimodal']['Test_F1'].values
trimodal = df[df['Type'] == 'Trimodal']['Test_F1'].values

data_to_plot = [unimodal, bimodal, trimodal]
labels = ['Unimodal\n(T, A, V)', 'Bimodal\n(TA, TV, AV)', 'Trimodal\n(TAV)']
colors_box = ['#2ecc71', '#3498db', '#e74c3c']

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6,
                showmeans=True, meanline=True,
                boxprops=dict(linewidth=2, edgecolor='black'),
                whiskerprops=dict(linewidth=2),
                capprops=dict(linewidth=2),
                medianprops=dict(linewidth=2.5, color='darkred'),
                meanprops=dict(linewidth=2.5, color='blue', linestyle='--'))

# Color boxes
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Test F1 Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Performance Distribution by Modality Type', fontsize=15, fontweight='bold', pad=20)
ax.set_ylim([60, 95])
ax.grid(axis='y', alpha=0.3)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='darkred', linewidth=2.5, label='Median'),
    Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='Mean')
]
ax.legend(handles=legend_elements, fontsize=11, loc='lower right')

# Add annotations for means
means = [np.mean(d) for d in data_to_plot]
for i, (pos, mean) in enumerate(zip([1, 2, 3], means)):
    ax.text(pos, mean + 1.5, f'μ={mean:.1f}%', ha='center', fontsize=10,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / 'modality_type_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'modality_type_comparison.png'}")
plt.close()

# ============================================================================
# PLOT 4: Text vs Emotions Comparison
# ============================================================================
print("Creating Plot 4: Text vs Emotions Direct Comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

# Compare Text vs Audio vs Video
comparison_data = df[df['Dataset'].isin(['unimodal_T', 'unimodal_A', 'unimodal_V'])]

x = np.arange(3)
metrics = ['Test_Accuracy', 'Test_F1', 'F1_Depression', 'F1_Mania', 'F1_Euthymia']
metric_names = ['Accuracy', 'F1 Score', 'F1\n(Depression)', 'F1\n(Mania)', 'F1\n(Euthymia)']

width = 0.25
colors_comp = ['#2ecc71', '#3498db', '#e74c3c']

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    values = comparison_data[metric].values
    offset = (i - 2) * width
    bars = ax.bar(x + offset, values, width, label=name, alpha=0.8, edgecolor='black', linewidth=1)

    # Color by modality
    for bar, color in zip(bars, colors_comp):
        bar.set_facecolor(color)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)

ax.set_xlabel('Modality', fontsize=13, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Text vs Emotion Features: Detailed Comparison', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(['Text\n(BERT)', 'Audio\n(6D Emotions)', 'Video\n(6D Emotions)'], fontsize=11)
ax.set_ylim([0, 100])
ax.legend(fontsize=10, loc='upper right', ncol=2)
ax.grid(axis='y', alpha=0.3)

# Add horizontal line for baseline
ax.axhline(y=48.8, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Majority Baseline')

plt.tight_layout()
plt.savefig(output_dir / 'text_vs_emotions.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'text_vs_emotions.png'}")
plt.close()

# ============================================================================
# PLOT 5: Heatmap of Per-Class Performance
# ============================================================================
print("Creating Plot 5: Per-Class Performance Heatmap...")

fig, ax = plt.subplots(figsize=(12, 8))

# Create matrix for heatmap
heatmap_data = df[['F1_Depression', 'F1_Mania', 'F1_Euthymia']].values.T
class_names = ['Depression', 'Mania', 'Euthymia']

# Create heatmap
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=50, vmax=95)

# Set ticks and labels
ax.set_xticks(np.arange(len(df)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(df['Short_Name'], fontsize=11)
ax.set_yticklabels(class_names, fontsize=12, fontweight='bold')

# Rotate the tick labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('F1 Score (%)', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(class_names)):
    for j in range(len(df)):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                      ha="center", va="center", color="black", fontsize=10, fontweight='bold')

ax.set_title('Per-Class F1 Scores Across Modality Combinations', fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Modality Combination', fontsize=13, fontweight='bold')
ax.set_ylabel('Class', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'per_class_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'per_class_heatmap.png'}")
plt.close()

# ============================================================================
# PLOT 6: Sample Size vs Performance
# ============================================================================
print("Creating Plot 6: Sample Size vs Performance Analysis...")

fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
scatter = ax.scatter(df['Train_Samples'], df['Test_F1'],
                    s=300, alpha=0.7, c=range(len(df)), cmap='viridis',
                    edgecolors='black', linewidth=2)

# Add labels for each point
for i, row in df.iterrows():
    ax.annotate(row['Short_Name'],
               (row['Train_Samples'], row['Test_F1']),
               fontsize=11, fontweight='bold', ha='center', va='center')

ax.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
ax.set_ylabel('Test F1 Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Training Sample Size vs Performance', fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

# Add annotation
ax.text(0.98, 0.02,
        'Note: Text-only (T) has more samples\nfrom Depression+Bipolar+MOSEI datasets',
        transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'samples_vs_performance.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'samples_vs_performance.png'}")
plt.close()

# ============================================================================
# PLOT 7: Key Findings Summary (Infographic Style)
# ============================================================================
print("Creating Plot 7: Key Findings Summary...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

# Finding 1: Text dominates
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
ax1.text(0.5, 0.8, 'KEY FINDING #1: TEXT DOMINATES',
         ha='center', fontsize=18, fontweight='bold', color='#2c3e50')
ax1.text(0.5, 0.5, 'Text (BERT): 87.0% F1',
         ha='center', fontsize=16, color='#27ae60', fontweight='bold')
ax1.text(0.5, 0.3, 'Emotions (6D): 72.6% F1',
         ha='center', fontsize=16, color='#3498db', fontweight='bold')
ax1.text(0.5, 0.1, 'Improvement: +14.4 percentage points',
         ha='center', fontsize=14, color='#e74c3c', fontweight='bold')
ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#2c3e50', linewidth=3))

# Finding 2: Data leakage fixed
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('off')
ax2.text(0.5, 0.9, 'FINDING #2', ha='center', fontsize=14, fontweight='bold', color='#2c3e50')
ax2.text(0.5, 0.7, 'Data Leakage Fixed!', ha='center', fontsize=13, fontweight='bold')
ax2.text(0.5, 0.5, 'Before: 99% (leaked)', ha='center', fontsize=12, color='#e74c3c')
ax2.text(0.5, 0.35, 'After: 72% (realistic)', ha='center', fontsize=12, color='#27ae60')
ax2.text(0.5, 0.15, '6D emotions only\n(no sentiment)', ha='center', fontsize=10, style='italic')
ax2.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#27ae60', linewidth=2))

# Finding 3: Emotions beat baseline
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.text(0.5, 0.9, 'FINDING #3', ha='center', fontsize=14, fontweight='bold', color='#2c3e50')
ax3.text(0.5, 0.7, 'Emotions Work!', ha='center', fontsize=13, fontweight='bold')
ax3.text(0.5, 0.5, 'Emotions: 72.6% F1', ha='center', fontsize=12, color='#3498db')
ax3.text(0.5, 0.35, 'Baseline: 48.8% F1', ha='center', fontsize=12, color='#95a5a6')
ax3.text(0.5, 0.15, 'Improvement: +23.8pp', ha='center', fontsize=11, fontweight='bold', color='#27ae60')
ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#3498db', linewidth=2))

# Finding 4: Audio = Video
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')
ax4.text(0.5, 0.9, 'FINDING #4', ha='center', fontsize=14, fontweight='bold', color='#2c3e50')
ax4.text(0.5, 0.7, 'Audio ≈ Video', ha='center', fontsize=13, fontweight='bold')
ax4.text(0.5, 0.5, 'Audio: 72.62% F1', ha='center', fontsize=12)
ax4.text(0.5, 0.35, 'Video: 72.64% F1', ha='center', fontsize=12)
ax4.text(0.5, 0.15, 'Same 6D emotions\n→ same performance', ha='center', fontsize=10, style='italic')
ax4.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#9b59b6', linewidth=2))

# Finding 5: Contribution
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
ax5.text(0.5, 0.9, 'CONTRIBUTION', ha='center', fontsize=14, fontweight='bold', color='#2c3e50')
ax5.text(0.5, 0.65, 'Text-based BD detection\noutperforms emotion features',
         ha='center', fontsize=11, fontweight='bold')
ax5.text(0.5, 0.35, '✓ Emotions work (72% F1)', ha='center', fontsize=10, color='#27ae60')
ax5.text(0.5, 0.2, '✓ But text is better (87% F1)', ha='center', fontsize=10, color='#27ae60')
ax5.text(0.5, 0.05, 'BD manifests in language\nmore than expressions',
         ha='center', fontsize=9, style='italic', color='#7f8c8d')
ax5.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#e74c3c', linewidth=2))

plt.savefig(output_dir / 'key_findings_summary.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'key_findings_summary.png'}")
plt.close()

# ============================================================================
# Create results summary JSON
# ============================================================================
print("\nCreating results summary JSON...")

summary = {
    'study_name': 'Ablation Study - No Data Leakage',
    'key_findings': {
        'finding_1': {
            'title': 'Text Dominates',
            'text_f1': 87.04,
            'emotions_f1': 72.62,
            'improvement': 14.42
        },
        'finding_2': {
            'title': 'Data Leakage Fixed',
            'before_accuracy': 99.0,
            'after_accuracy': 72.6,
            'fix': 'Excluded sentiment from 6D emotion features'
        },
        'finding_3': {
            'title': 'Emotions Beat Baseline',
            'emotions_f1': 72.6,
            'baseline_accuracy': 48.8,
            'improvement': 23.8
        },
        'finding_4': {
            'title': 'Audio ≈ Video',
            'audio_f1': 72.62,
            'video_f1': 72.64,
            'explanation': 'Same 6D emotion features'
        }
    },
    'best_models': {
        'overall_best': {
            'modality': 'Text-only (unimodal_T)',
            'f1': 87.04,
            'accuracy': 88.13
        },
        'best_emotions': {
            'modality': 'Video-only (unimodal_V)',
            'f1': 72.64,
            'accuracy': 72.91
        },
        'best_multimodal': {
            'modality': 'Trimodal (TAV)',
            'f1': 73.05,
            'accuracy': 73.32,
            'note': 'Text features were zeros - really just A+V'
        }
    },
    'per_class_performance': {
        'text_only': {
            'depression': 82.96,
            'mania': 93.35,
            'euthymia': 54.92
        },
        'emotions': {
            'depression': 60.87,
            'mania': 75.19,
            'euthymia': 73.47
        }
    }
}

with open(output_dir / 'results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  ✓ Saved: {output_dir / 'results_summary.json'}")

print()
print("="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nAll visualizations saved to: {output_dir}/")
print("\nGenerated files:")
print("  1. overall_performance.png - Overall accuracy & F1 comparison")
print("  2. per_class_f1.png - Per-class F1 scores grouped bar chart")
print("  3. modality_type_comparison.png - Unimodal vs Bimodal vs Trimodal")
print("  4. text_vs_emotions.png - Direct comparison of text vs emotions")
print("  5. per_class_heatmap.png - Heatmap of per-class performance")
print("  6. samples_vs_performance.png - Training size vs performance")
print("  7. key_findings_summary.png - Infographic of key findings")
print("  8. results_summary.json - Structured summary of results")
print()
print("Use these visualizations in your paper!")
print("="*80)
