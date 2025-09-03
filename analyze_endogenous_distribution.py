#!/usr/bin/env python3
"""
Analyze DeepSTARR score distribution for endogenous sequences
Recreates activity distribution analysis similar to Figure 2B/C setup
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

def load_filtered_data():
    """Load the summary statistics data"""
    print("Loading summary data...")
    
    with open('DeepSTARR_dm6_summary_statistics.pkl', 'rb') as f:
        summary_df = pickle.load(f)
    
    print(f"Original data: {len(summary_df)} total sequences")
    print(f"Original motif pairs: {sorted(summary_df['motif_pair'].unique())}")
    
    # Filter out high-diversity pairs
    high_diversity_pairs = ['twi_twi', 'GATAe_GATAe']
    summary_df_filtered = summary_df[~summary_df['motif_pair'].isin(high_diversity_pairs)].copy()
    
    print(f"After filtering out {high_diversity_pairs}: {len(summary_df_filtered)} sequences")
    print(f"Motif pairs included: {sorted(summary_df_filtered['motif_pair'].unique())}")
    
    # Add activity bin labels for both dev and hk scores
    print("Adding activity bin labels...")
    
    # Add dev activity bins
    dev_bins = create_activity_bins_from_scores(summary_df_filtered['endogenous_dev_score'])
    summary_df_filtered['dev_activity_bin'] = dev_bins
    
    # Add hk activity bins  
    hk_bins = create_activity_bins_from_scores(summary_df_filtered['endogenous_hk_score'])
    summary_df_filtered['hk_activity_bin'] = hk_bins
    
    # Print distribution of bins
    dev_counts = pd.Series(dev_bins).value_counts()
    hk_counts = pd.Series(hk_bins).value_counts()
    
    print(f"DEV activity bins: {dict(dev_counts)}")
    print(f"HK activity bins: {dict(hk_counts)}")
    
    print(f"Final dataframe columns: {list(summary_df_filtered.columns)}")
    
    return summary_df_filtered

def create_activity_bins_from_scores(scores, n_bins=3):
    """Create activity bins (high, medium, low) based on score distribution"""
    # Extract scores from lists if needed
    if hasattr(scores.iloc[0], '__len__') and not isinstance(scores.iloc[0], str):
        score_values = np.array([score[0] if hasattr(score, '__len__') else score for score in scores])
    else:
        score_values = scores.values
    
    score_values = score_values.astype(float)
    
    # Create quantile-based bins
    if n_bins == 3:
        low_thresh = np.percentile(score_values, 33.33)
        high_thresh = np.percentile(score_values, 66.67)
        
        conditions = [
            score_values <= low_thresh,
            (score_values > low_thresh) & (score_values <= high_thresh),
            score_values > high_thresh
        ]
        choices = ['Low', 'Medium', 'High']
    
    activity_bins = np.select(conditions, choices)
    
    return activity_bins

def plot_score_distributions(endogenous_df):
    """Plot the distribution of DeepSTARR scores for endogenous sequences"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, score_type in enumerate(['dev', 'hk']):
        score_col = f'endogenous_{score_type}_score'
        bin_col = f'{score_type}_activity_bin'
        score_name = 'Developmental' if score_type == 'dev' else 'Housekeeping'
        
        # Extract scores and handle potential array/list issues
        score_series = endogenous_df[score_col]
        
        # Check if scores are stored as lists/arrays
        if hasattr(score_series.iloc[0], '__len__') and not isinstance(score_series.iloc[0], str):
            # If scores are stored as arrays/lists, extract the first element
            scores = np.array([score[0] if hasattr(score, '__len__') else score for score in score_series])
        else:
            scores = score_series.values
        
        scores = scores.astype(float)
        
        print(f"  {score_name} scores shape: {scores.shape}, type: {type(scores[0])}")
        
        # Use pre-computed activity bins
        activity_bins = endogenous_df[bin_col].values
        
        # Plot 1: Overall distribution
        ax1 = axes[i, 0]
        ax1.hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(np.percentile(scores, 33.33), color='red', linestyle='--', alpha=0.7, label='33rd percentile')
        ax1.axvline(np.percentile(scores, 66.67), color='red', linestyle='--', alpha=0.7, label='67th percentile')
        ax1.set_xlabel(f'{score_name} Score', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title(f'{score_name} Score Distribution\n(Endogenous Sequences)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Binned distribution
        ax2 = axes[i, 1]
        
        # Create dataframe for plotting
        plot_df = endogenous_df.copy()
        plot_df['score'] = scores
        
        # Box plot by activity bin
        bin_order = ['Low', 'Medium', 'High']
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        box_data = [plot_df[plot_df[bin_col] == bin_name]['score'].values 
                   for bin_name in bin_order]
        
        # Create boxplot without outliers (we'll add all points manually)
        bp = ax2.boxplot(box_data, labels=bin_order, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add all individual points with jitter for each bin
        np.random.seed(42)  # For reproducible jitter
        for j, bin_name in enumerate(bin_order):
            bin_scores = plot_df[plot_df[bin_col] == bin_name]['score'].values
            if len(bin_scores) > 0:
                # Add jitter to x-position
                x_pos = np.random.normal(j+1, 0.04, len(bin_scores))
                ax2.scatter(x_pos, bin_scores, color='black', alpha=0.6, s=20, zorder=3)
        
        ax2.set_xlabel('Activity Bin', fontsize=12)
        ax2.set_ylabel(f'{score_name} Score', fontsize=12)
        ax2.set_title(f'{score_name} Scores by Activity Bin\n(All Individual Points Shown)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add count annotations
        for j, bin_name in enumerate(bin_order):
            count = len(plot_df[plot_df[bin_col] == bin_name])
            ax2.text(j+1, ax2.get_ylim()[1] * 0.95, f'n={count}', 
                    ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('endogenous_score_distributions.svg', format='svg', bbox_inches='tight')
    print("Saved endogenous score distributions as endogenous_score_distributions.svg")
    plt.show()

def plot_motif_pair_distributions(endogenous_df):
    """Plot score distributions by motif pair"""
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    motif_pairs = sorted(endogenous_df['motif_pair'].unique())
    
    for i, score_type in enumerate(['dev', 'hk']):
        score_col = f'endogenous_{score_type}_score'
        score_name = 'Developmental' if score_type == 'dev' else 'Housekeeping'
        
        ax = axes[i]
        
        # Create violin plot
        plot_data = []
        plot_labels = []
        
        for motif_pair in motif_pairs:
            pair_data = endogenous_df[endogenous_df['motif_pair'] == motif_pair]
            
            # Handle score extraction like in the main plot
            score_series = pair_data[score_col]
            if hasattr(score_series.iloc[0], '__len__') and not isinstance(score_series.iloc[0], str):
                scores = np.array([score[0] if hasattr(score, '__len__') else score for score in score_series])
            else:
                scores = score_series.values
            
            scores = scores.astype(float)
            
            # Only include if we have more than 1 data point for violin plot
            if len(scores) > 1:
                plot_data.append(scores)
                plot_labels.append(f"{motif_pair}\n(n={len(scores)})")
            else:
                print(f"  Skipping {motif_pair} - only {len(scores)} data point(s)")
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No motif pairs with >1 data point', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{score_name} Score Distribution by Motif Pair\n(Insufficient data)', fontsize=14)
            continue
        
        # Violin plot
        parts = ax.violinplot(plot_data, positions=range(len(plot_data)), showmeans=True)
        
        # Color the violins
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
        ax.set_ylabel(f'{score_name} Score', fontsize=12)
        ax.set_title(f'{score_name} Score Distribution by Motif Pair\n(Endogenous Sequences)', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('endogenous_scores_by_motif_pair.svg', format='svg', bbox_inches='tight')
    print("Saved motif pair distributions as endogenous_scores_by_motif_pair.svg")
    plt.show()

def print_summary_stats(endogenous_df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("ENDOGENOUS SEQUENCE ANALYSIS SUMMARY")
    print("="*60)
    
    for score_type in ['dev', 'hk']:
        score_col = f'endogenous_{score_type}_score'
        bin_col = f'{score_type}_activity_bin'
        score_name = 'Developmental' if score_type == 'dev' else 'Housekeeping'
        
        # Handle score extraction like in the plots
        score_series = endogenous_df[score_col]
        if hasattr(score_series.iloc[0], '__len__') and not isinstance(score_series.iloc[0], str):
            scores = np.array([score[0] if hasattr(score, '__len__') else score for score in score_series])
        else:
            scores = score_series.values
        scores = scores.astype(float)
        
        print(f"\n{score_name} Scores:")
        print(f"  Total sequences: {len(scores)}")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Std: {np.std(scores):.4f}")
        print(f"  Min: {np.min(scores):.4f}")
        print(f"  Max: {np.max(scores):.4f}")
        print(f"  Median: {np.median(scores):.4f}")
        
        # Use pre-computed activity bins
        activity_bins = endogenous_df[bin_col].values
        bin_names = ['Low', 'Medium', 'High']
        print(f"  Activity bin distribution:")
        for bin_name in bin_names:
            count = np.sum(activity_bins == bin_name)
            percentage = 100 * count / len(scores)
            print(f"    {bin_name}: {count} sequences ({percentage:.1f}%)")
    
    print(f"\nMotif pair distribution:")
    motif_counts = endogenous_df['motif_pair'].value_counts()
    for motif_pair, count in motif_counts.items():
        percentage = 100 * count / len(endogenous_df)
        print(f"  {motif_pair}: {count} sequences ({percentage:.1f}%)")

def main():
    """Main analysis function"""
    print("Analyzing endogenous sequence DeepSTARR score distributions...")
    print("="*60)
    
    # Load data
    endogenous_df = load_filtered_data()
    
    # Create distribution plots
    print("\n1. Creating score distribution plots...")
    plot_score_distributions(endogenous_df)
    
    print("\n2. Creating motif pair distribution plots...")
    plot_motif_pair_distributions(endogenous_df)
    
    # Print summary statistics
    print_summary_stats(endogenous_df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("Generated files:")
    print("- endogenous_score_distributions.svg")
    print("- endogenous_scores_by_motif_pair.svg")

if __name__ == "__main__":
    main() 