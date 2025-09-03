#!/usr/bin/env python3
"""
Comprehensive CE Analysis Pipeline
- Generates 30 natural backgrounds per CE sequence
- Runs DeepSTARR scoring on all sequences
- Creates comparison plots with statistical tests
"""

import ceseek
import math
import pandas as pd
from statsmodels.stats.multitest import multipletests
from itertools import product
import tangermeme
import random
import torch
from tangermeme.utils import one_hot_encode, characters
from tangermeme.ersatz import substitute, dinucleotide_shuffle

# DeepSTARR imports
import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, model_from_json
from urllib.request import urlretrieve

# Plotting imports
import matplotlib.pyplot as plt
from scipy import stats
import pickle

# Configuration
BACKGROUNDS_PER_CE_SEQUENCE = 30
RANDOM_SEED = 33

def initialize_ceseek(pwm_file, genome_file, atac_file, control_file):
    obj = ceseek.CEseek(pwm_file)
    obj.load_sequences(atac_file, genome_sequence_fa=genome_file, sequence_set='target', out_seq_len=250)
    obj.load_sequences(atac_file, genome_sequence_fa=control_file, sequence_set='control', out_seq_len=250)
    obj.scan_motifs(num_threads=8)
    return obj

def run_enrichment_analysis():
    """Run the initial enrichment analysis to find significant CEs"""
    pwm_file = 'DeepSTARR_dm6_motifs.pwm'
    genome_file = 'filtered_dm6_genome_simple.fasta'
    atac_file = 'atac_dataset1_regions_clean.bed'
    control_file = 'filtered_dm6_dishuffled_genome.fasta'

    obj = initialize_ceseek(pwm_file, genome_file, atac_file, control_file)

    print("=== Enrichment Analysis ===")
    print(f"Loading motifs from {pwm_file}...contains {len(obj.dict_motif.keys())} motifs")
    print(f"Loading genome from {genome_file}...")
    print(f"Loading atac file from {atac_file}...")
    print(f"Loading control file from {control_file}...")

    motif_names = list(obj.dict_motif.keys())
    motif_pairs = list(product(motif_names, repeat=2))
    print(f"\nAnalyzing {len(motif_pairs)} motif pairs...")
    
    # Store results for DataFrame
    results = []
    total_target = len(obj.sequences)
    total_control = len(obj.sequences_control)

    for motif1, motif2 in motif_pairs:
        target_counts = obj.count_CE_configuration_for_motif_pair(
            motif1, motif2, target_sequences=True, device='cuda'
        )
        control_counts = obj.count_CE_configuration_for_motif_pair(
            motif1, motif2, target_sequences=False, device='cuda'
        )

        for config_pair in target_counts.keys():
            for strand_pair in target_counts[config_pair].keys():
                for spacing, count_target in target_counts[config_pair][strand_pair].items():
                    count_control = control_counts[config_pair][strand_pair][spacing]

                    a = count_target
                    A = total_target
                    b = count_control
                    B = total_control

                    p_val = obj._fisher_exact_test_for_motif_hits(a, A, b, B)
                    log2_fc = math.log2(((count_target + 1e-9) / total_target) / ((count_control + 1e-9) / total_control))

                    # Only extract sequences for significant results with reasonable counts
                    if p_val < 0.05 and count_target <= 500:
                        # Extract CE sequences and their genomic contexts
                        config = config_pair
                        strand1, strand2 = strand_pair
                        
                        # Use CE_name_standardization to get the standardized names that extract_CE_sequences expects
                        preferred_CE_name, other_CE_name = obj.CE_name_standardization(motif1, motif2, strand1, strand2)
                        std_motif1, std_motif2, std_strand1, std_strand2 = preferred_CE_name
                        
                        # Get CE sequences, pseudo-indices, and 249bp extended sequences using standardized names
                        ce_sequences, pseudo_indices, genomic_windows = obj.extract_CE_sequences(
                            std_motif1, std_motif2, std_strand1, std_strand2, spacing, 
                            device='cuda', return_pseudo_indices=True
                        )
                        
                        motif1_len = obj.dict_motif[motif1].shape[0]
                        motif2_len = obj.dict_motif[motif2].shape[0]
                        ce_length = motif1_len + spacing + motif2_len

                        motif_locations = []
                        for i, (pseudo_idx, orientation) in enumerate(pseudo_indices):
                            # Calculate the genomic window start for this specific CE instance
                            seq_ind = pseudo_idx // (obj.sequence_length * 2)
                            seq_loc = pseudo_idx % (obj.sequence_length * 2)
                            
                            peak_seq = obj.sequences[seq_ind]
                            peak_len = len(peak_seq)
                            
                            # Calculate 249bp window boundaries (same logic as extract_CE_sequences)
                            pad_total = 249 - ce_length
                            pad_left = min(pad_total // 2, seq_loc)
                            pad_right = min(pad_total - pad_left, peak_len - (seq_loc + ce_length))
                            
                            genomic_window_start = max(0, seq_loc - pad_left)
                            genomic_window_end = min(peak_len, seq_loc + ce_length + pad_right)
                            
                            if genomic_window_end - genomic_window_start < 249:
                                diff = 249 - (genomic_window_end - genomic_window_start)
                                if genomic_window_end < peak_len:
                                    genomic_window_end += diff
                                else:
                                    genomic_window_start -= diff
                            
                            # Convert pseudo-genome coordinates to sequence-local coordinates (within 249bp window)
                            motif1_start = seq_loc - genomic_window_start
                            motif1_end = motif1_start + motif1_len
                            motif2_start = motif1_end + spacing
                            motif2_end = motif2_start + motif2_len

                            motif_locations.append({
                                "motif1_start": motif1_start,
                                "motif1_end": motif1_end,
                                "motif2_start": motif2_start,
                                "motif2_end": motif2_end,
                                "orientation": orientation
                            })

                        results.append({
                            'motif1': std_motif1,
                            'motif2': std_motif2,
                            'config': f"{config_pair}_{strand_pair}_{spacing}",
                            'ce_count_target': count_target,
                            'ce_count_control': count_control,
                            'log2_fc': log2_fc,
                            'p_value': p_val,
                            'ce_sequences': ce_sequences,
                            'genomic_windows': genomic_windows,
                            'motif_locations': motif_locations
                        })

    # Benjamini–Hochberg FDR correction (q-values) via statsmodels
    pvals = [r['p_value'] for r in results]
    if len(pvals) > 0:
        _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
        for r, q in zip(results, qvals):
            r['q_value'] = float(min(q, 1.0))

    df = pd.DataFrame(results)
    df = df[df['ce_sequences'].apply(len) > 0]
    df = df.sort_values(by='p_value', ascending=True)

    # Remove duplicate configurations that arise from reciprocal motif pairs
    # Group by config and keep only the first occurrence (they have identical sequences)
    print(f"Before deduplication: {len(df)} significant CE configurations")
    
    # Create a unique identifier for each configuration based on actual content
    df['config_id'] = df['config'] + '_' + df['motif1'] + '_' + df['motif2']
    
    # For each config, check if there are duplicates with same sequences
    unique_configs = []
    seen_configs = set()
    
    for idx, row in df.iterrows():
        config_signature = (row['config'], tuple(row['ce_sequences']), tuple(row['genomic_windows']))
        if config_signature not in seen_configs:
            seen_configs.add(config_signature)
            unique_configs.append(idx)
    
    df = df.loc[unique_configs]
    df = df.drop('config_id', axis=1)  # Remove temporary column
    
    print(f"After deduplication: {len(df)} unique CE configurations")
    return df

def generate_multiple_backgrounds(df):
    """Generate 30 natural backgrounds per CE sequence"""
    print(f"=== Generating {BACKGROUNDS_PER_CE_SEQUENCE} Natural Backgrounds Per CE Sequence ===")
    
    # Set random seed for reproducible background selection
    random.seed(RANDOM_SEED)
    
    # Create expanded results list
    expanded_results = []
    
    for idx, row in df.iterrows():
        current_motif1 = row['motif1']
        current_motif2 = row['motif2']
        
        # Collect all available background options from CEs with different motifs
        all_bg_options = []
        for other_idx, other_row in df.iterrows():
            if other_idx != idx:  # Skip same CE
                other_motif1 = other_row['motif1']
                other_motif2 = other_row['motif2']
                
                # Check if motifs are different
                # For homotypic CEs (motif1 == motif2), only exclude same homotypic pairs
                # For heterotypic CEs, exclude any CE containing either motif
                if current_motif1 == current_motif2:  # Homotypic CE
                    # Only exclude other homotypic CEs with the same motif
                    if not (other_motif1 == current_motif1 and other_motif2 == current_motif2):
                        include_background = True
                    else:
                        include_background = False
                else:  # Heterotypic CE
                    # Exclude any CE containing either of our motifs
                    if (other_motif1 != current_motif1 and other_motif1 != current_motif2 and
                        other_motif2 != current_motif1 and other_motif2 != current_motif2):
                        include_background = True
                    else:
                        include_background = False
                
                if include_background:
                    
                    for window_idx in range(len(other_row['genomic_windows'])):
                        all_bg_options.append({
                            'config': other_row['config'],
                            'df_idx': other_idx,
                            'window_idx': window_idx,
                            'motif1': other_motif1,
                            'motif2': other_motif2,
                            'sequence': other_row['genomic_windows'][window_idx],
                            'motif_locations': other_row['motif_locations'][window_idx]
                        })
        
        #print(f"CE {row['config']} ({current_motif1}_{current_motif2}): {len(all_bg_options)} background options")
        
        # For each CE sequence, create 30 natural backgrounds
        for ce_seq_idx, (ce_seq, endo_window) in enumerate(zip(row['ce_sequences'], row['genomic_windows'])):
            
            # Select 30 random backgrounds (with replacement if needed)
            if len(all_bg_options) >= BACKGROUNDS_PER_CE_SEQUENCE:
                selected_bgs = random.sample(all_bg_options, BACKGROUNDS_PER_CE_SEQUENCE)
            else:
                selected_bgs = random.choices(all_bg_options, k=BACKGROUNDS_PER_CE_SEQUENCE)
            
            natural_bg_sequences = []
            
            for bg_idx, bg_option in enumerate(selected_bgs):
                bg_sequence = bg_option['sequence']
                bg_motif_loc = bg_option['motif_locations']
                bg_motif1_start = bg_motif_loc['motif1_start']
                bg_motif2_end = bg_motif_loc['motif2_end']
                
                # Try to place CE at the background's motif position
                attempt_count = 0
                max_attempts = 50
                
                while bg_motif1_start + len(ce_seq) > len(bg_sequence) and attempt_count < max_attempts:
                    # Select a different background if current one doesn't fit
                    bg_option = random.choice(all_bg_options)
                    bg_sequence = bg_option['sequence']
                    bg_motif_loc = bg_option['motif_locations']
                    bg_motif1_start = bg_motif_loc['motif1_start']
                    bg_motif2_end = bg_motif_loc['motif2_end']
                    attempt_count += 1
                
                if attempt_count >= max_attempts:
                    print(f"Warning: Could not find suitable background for CE {row['config']} sequence {ce_seq_idx}")
                    continue
                
                # Dinucleotide shuffle the original CE region in the background
                bg_ohe = one_hot_encode(bg_sequence.upper())
                original_ce_region = bg_sequence[bg_motif1_start:bg_motif2_end]
                
                if len(original_ce_region) > 0:
                    original_region_ohe = one_hot_encode(original_ce_region.upper())
                    shuffled_region_ohe = dinucleotide_shuffle(original_region_ohe.unsqueeze(0), n=1)
                    
                    # Ensure correct dimensions for characters()
                    while shuffled_region_ohe.dim() > 2:
                        shuffled_region_ohe = shuffled_region_ohe.squeeze(0)
                    
                    shuffled_region_seq = characters(shuffled_region_ohe)
                    
                    # Replace the original CE region with the shuffled version
                    bg_with_shuffled = (bg_sequence[:bg_motif1_start] + 
                                      shuffled_region_seq + 
                                      bg_sequence[bg_motif2_end:])
                    bg_ohe = one_hot_encode(bg_with_shuffled.upper())
                
                # Substitute the new CE sequence at the motif1_start position
                natural_background_ohe = substitute(bg_ohe.unsqueeze(0), ce_seq.upper(), start=bg_motif1_start)
                natural_background_seq = characters(natural_background_ohe[0])
                
                # Store sequence with source metadata
                natural_bg_sequences.append({
                    'sequence': natural_background_seq,
                    'source_config': bg_option['config'],
                    'source_df_idx': bg_option['df_idx'],
                    'source_window_idx': bg_option['window_idx'],
                    'source_motif1': bg_option['motif1'],
                    'source_motif2': bg_option['motif2']
                })
            
            # Create individual result entries for this CE sequence
            expanded_results.append({
                'config': row['config'],
                'motif1': row['motif1'],
                'motif2': row['motif2'],
                'motif_pair': f"{row['motif1']}_{row['motif2']}",
                'ce_sequence_index': ce_seq_idx,
                'ce_sequence': ce_seq,
                'endogenous_sequence': endo_window,
                'natural_background_sequences': natural_bg_sequences,
                'num_backgrounds': len(natural_bg_sequences)
            })
    
    expanded_df = pd.DataFrame(expanded_results)
    print(f"Generated {len(expanded_df)} CE sequence entries with {BACKGROUNDS_PER_CE_SEQUENCE} backgrounds each")
    return expanded_df

def setup_deepstarr():
    """Download and load DeepSTARR model"""
    print("=== Setting up DeepSTARR Model ===")
    
    assets_dir = 'assets_deepstarr'
    os.makedirs(assets_dir, exist_ok=True)

    def download_if_not_exists(url, filename):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urlretrieve(url, filename)
        else:
            print(f"Using existing {filename}")

    files = {
        'deepstarr.model.json': 'https://www.dropbox.com/scl/fi/y1mwsqpv2e514md9t68jz/deepstarr.model.json?rlkey=cdwhstqf96fibshes2aov6t1e&st=9a0c5skz&dl=1',
        'deepstarr.model.h5': 'https://www.dropbox.com/scl/fi/6nl6e2hofyw70lh99h3uk/deepstarr.model.h5?rlkey=hqfnivn199xa54bjh8dn2jpaf&st=l4jig4ky&dl=1',
        'deepstarr_data.h5': 'https://www.dropbox.com/scl/fi/cya4ntqk2o8yftxql52lu/deepstarr_data.h5?rlkey=5ly363vqjb3vaw2euw2dhsjo3&st=6eod6fg8&dl=1'
    }

    # Download files to assets_deepstarr folder
    for filename, url in files.items():
        filepath = os.path.join(assets_dir, filename)
        download_if_not_exists(url, filepath)

    keras_model_weights = os.path.join(assets_dir, 'deepstarr.model.h5')
    keras_model_json = os.path.join(assets_dir, 'deepstarr.model.json')

    # Load model
    keras_model = model_from_json(open(keras_model_json).read(), custom_objects={'Functional': tf.keras.Model})
    np.random.seed(113)
    random.seed(0)
    keras_model.load_weights(keras_model_weights)
    
    print("DeepSTARR model loaded successfully")
    return keras_model

def score_sequences(expanded_df, model):
    """Score all sequences with DeepSTARR"""
    print("=== Scoring Sequences with DeepSTARR ===")
    
    # Collect all sequences and metadata
    all_sequences = []
    metadata = []
    
    for idx, row in expanded_df.iterrows():
        config = row['config']
        motif1 = row['motif1']
        motif2 = row['motif2']
        motif_pair = row['motif_pair']
        ce_seq_idx = row['ce_sequence_index']
        
        # Add endogenous sequence
        endo_seq = row['endogenous_sequence']
        all_sequences.append(endo_seq)
        metadata.append({
            'config': config, 'motif1': motif1, 'motif2': motif2, 'motif_pair': motif_pair,
            'ce_sequence_index': ce_seq_idx, 'sequence_type': 'endogenous', 'sequence': endo_seq
        })
        
        # Add background sequences
        for bg_idx, bg_data in enumerate(row['natural_background_sequences']):
            bg_seq = bg_data['sequence']
            all_sequences.append(bg_seq)
            metadata.append({
                'config': config, 'motif1': motif1, 'motif2': motif2, 'motif_pair': motif_pair,
                'ce_sequence_index': ce_seq_idx, 'sequence_type': 'natural_background',
                'background_index': bg_idx, 'sequence': bg_seq,
                'source_config': bg_data['source_config'],
                'source_df_idx': bg_data['source_df_idx'],
                'source_window_idx': bg_data['source_window_idx'],
                'source_motif1': bg_data['source_motif1'],
                'source_motif2': bg_data['source_motif2']
            })
    
    # Batch process all sequences
    print(f"Processing {len(all_sequences)} sequences in one batch...")
    all_ohe = [one_hot_encode(seq.upper()).numpy().T for seq in all_sequences]
    all_pred = model.predict(np.array(all_ohe), verbose=0)
    
    # Combine results
    individual_results = []
    for i, meta in enumerate(metadata):
        result = meta.copy()
        result['dev_score'] = all_pred[0][i]
        result['hk_score'] = all_pred[1][i]
        individual_results.append(result)
    
    individual_df = pd.DataFrame(individual_results)
    print(f"Scored {len(individual_df)} individual sequences")
    
    return individual_df

def create_summary_statistics(individual_df):
    """Create summary statistics comparing endogenous vs natural backgrounds"""
    print("=== Creating Summary Statistics ===")
    
    summary_results = []
    
    # Group by config and ce_sequence_index
    for (config, ce_seq_idx), group in individual_df.groupby(['config', 'ce_sequence_index']):
        endo_data = group[group['sequence_type'] == 'endogenous']
        natbg_data = group[group['sequence_type'] == 'natural_background']
        
        if len(endo_data) == 1 and len(natbg_data) > 0:
            # Get single endogenous scores
            endo_dev = endo_data['dev_score'].iloc[0]
            endo_hk = endo_data['hk_score'].iloc[0]
            
            # Get natural background scores and ensure proper numeric types
            natbg_dev_scores = natbg_data['dev_score'].values.astype(float)
            natbg_hk_scores = natbg_data['hk_score'].values.astype(float)
            
            # Calculate p-values using one-sample t-tests
            if len(natbg_dev_scores) > 1:
                _, dev_pval = stats.ttest_1samp(natbg_dev_scores, float(endo_dev))
            else:
                dev_pval = 1.0
                
            if len(natbg_hk_scores) > 1:
                _, hk_pval = stats.ttest_1samp(natbg_hk_scores, float(endo_hk))
            else:
                hk_pval = 1.0
            
            summary_results.append({
                'config': config,
                'motif1': endo_data['motif1'].iloc[0],
                'motif2': endo_data['motif2'].iloc[0],
                'motif_pair': endo_data['motif_pair'].iloc[0],
                'ce_sequence_index': ce_seq_idx,
                'num_backgrounds': len(natbg_data),
                
                # Endogenous scores (single values)
                'endogenous_dev_score': endo_dev,
                'endogenous_hk_score': endo_hk,
                
                # Natural background statistics
                'natural_background_dev_mean': np.mean(natbg_dev_scores),
                'natural_background_dev_std': np.std(natbg_dev_scores),
                'natural_background_hk_mean': np.mean(natbg_hk_scores),
                'natural_background_hk_std': np.std(natbg_hk_scores),
                
                # Differences (endogenous vs mean of backgrounds)
                'dev_diff': endo_dev - np.mean(natbg_dev_scores),
                'hk_diff': endo_hk - np.mean(natbg_hk_scores),
                
                # P-values from statistical tests
                'dev_pval': dev_pval,
                'hk_pval': hk_pval
            })
    
    summary_df = pd.DataFrame(summary_results)
    print(f"Created summary for {len(summary_df)} CE sequence instances")
    
    return summary_df

def create_plots(individual_df, summary_df):
    """Create comparison plots"""
    print("=== Creating Plots ===")
    
    def create_comparison_plot(score_type, output_filename):
        if score_type == 'dev':
            score_col = 'dev_score'
            diff_col = 'dev_diff'
            score_name = 'Developmental'
        else:
            score_col = 'hk_score'
            diff_col = 'hk_diff'
            score_name = 'Housekeeping'
        
        # Sort by differences for visualization
        df_sorted = summary_df.sort_values(diff_col, ascending=False)
        
        # Calculate grid size
        n_entries = len(df_sorted)
        n_cols = 10
        n_rows = int(np.ceil(n_entries / n_cols))
        
        # Create plot with better spacing
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(35, 5*n_rows))
        axes = axes.flatten()
        
        significant_count = 0
        
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            ax = axes[i]
            
            # Get scores for this specific CE sequence instance
            ce_data = individual_df[
                (individual_df['config'] == row['config']) & 
                (individual_df['ce_sequence_index'] == row['ce_sequence_index'])
            ]
            
            endo_score = float(ce_data[ce_data['sequence_type'] == 'endogenous'][score_col].iloc[0])
            natbg_scores = ce_data[ce_data['sequence_type'] == 'natural_background'][score_col].values.astype(float)
            
            # Perform t-test (one-sample t-test comparing backgrounds to endogenous)
            if len(natbg_scores) > 1:
                t_stat, p_val = stats.ttest_1samp(natbg_scores, endo_score)
            else:
                p_val = 1.0
            
            is_significant = p_val < 0.05
            if is_significant:
                significant_count += 1
            
            # Generate a unique color for this endogenous-background group
            # Use config and ce_sequence_index to create a consistent color
            color_seed = hash(f"{row['config']}_{row['ce_sequence_index']}") % 1000
            np.random.seed(color_seed)
            group_color = plt.cm.tab10(np.random.rand())
            
            # Plot individual background points with jitter
            np.random.seed(42)
            x_natbg = np.random.normal(1, 0.05, len(natbg_scores))
            
            # Plot endogenous as single point with group color
            ax.scatter([0], [endo_score], color=group_color, alpha=0.9, s=50, marker='D', edgecolors='black', linewidth=1)
            ax.scatter(x_natbg, natbg_scores, color=group_color, alpha=0.7, s=30)
            
            # Add mean line for backgrounds with group color
            ax.axhline(y=np.mean(natbg_scores), xmin=0.6, xmax=1.0, color=group_color, linewidth=3, alpha=0.8)
            
            # Formatting
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Endo', 'NatBG'], fontsize=12)
            ax.set_ylabel(f'{score_name} Score', fontsize=12)
            
            # Create title
            sig_marker = '*' if is_significant else ''
            diff = float(row[diff_col])
            title_text = f"{row['config']}_seq{row['ce_sequence_index']}{sig_marker} diff={diff:.3f} p={float(p_val):.3f}"
            ax.set_title(title_text, fontsize=10, pad=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_xlim(-0.3, 1.3)
        
        # Hide unused subplots
        for i in range(n_entries, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'All CE Sequences - {score_name} Scores (30 Backgrounds Each)' + ' ' +
                     f'{significant_count}/{n_entries} sequences with significant differences (p<0.05)', 
                     fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save as SVG for perfect zooming
        svg_filename = output_filename.replace('.png', '.svg')
        plt.savefig(svg_filename, format='svg', bbox_inches='tight')
        print(f"Saved as {svg_filename} (vector format - perfect zoom)")
        
        plt.show()
        
        print(f'{score_name} plot saved as {output_filename}')
        print(f'Significant sequences: {significant_count}/{n_entries} ({100*significant_count/n_entries:.1f}%)')
    
    def create_focused_plots(individual_df, summary_df):
        """Create focused plots with top significant results for better viewing"""
        print("Creating focused plots with top results...")
        
        for score_type in ['dev', 'hk']:
            if score_type == 'dev':
                score_col = 'dev_score'
                diff_col = 'dev_diff'
                score_name = 'Developmental'
            else:
                score_col = 'hk_score'
                diff_col = 'hk_diff'
                score_name = 'Housekeeping'
            
            # Get top 20 most significant results
            df_sorted = summary_df.sort_values(diff_col, ascending=False).head(20)
            
            # Create 4x5 grid for top 20
            fig, axes = plt.subplots(4, 5, figsize=(20, 16))
            axes = axes.flatten()
            
            for i, (idx, row) in enumerate(df_sorted.iterrows()):
                ax = axes[i]
                
                # Get scores for this CE sequence
                ce_data = individual_df[
                    (individual_df['config'] == row['config']) & 
                    (individual_df['ce_sequence_index'] == row['ce_sequence_index'])
                ]
                
                endo_score = float(ce_data[ce_data['sequence_type'] == 'endogenous'][score_col].iloc[0])
                natbg_scores = ce_data[ce_data['sequence_type'] == 'natural_background'][score_col].values.astype(float)
                
                # Statistical test
                if len(natbg_scores) > 1:
                    t_stat, p_val = stats.ttest_1samp(natbg_scores, endo_score)
                else:
                    p_val = 1.0
                
                # Generate a unique color for this endogenous-background group
                color_seed = hash(f"{row['config']}_{row['ce_sequence_index']}") % 1000
                np.random.seed(color_seed)
                group_color = plt.cm.tab10(np.random.rand())
                
                # Plot with larger elements for better visibility
                np.random.seed(42)
                x_natbg = np.random.normal(1, 0.05, len(natbg_scores))
                
                ax.scatter([0], [endo_score], color=group_color, alpha=0.9, s=100, marker='D', 
                          label='Endogenous' if i == 0 else '', edgecolors='black', linewidth=1)
                ax.scatter(x_natbg, natbg_scores, color=group_color, alpha=0.7, s=50, 
                          label='Natural BG' if i == 0 else '')
                ax.axhline(y=np.mean(natbg_scores), xmin=0.6, xmax=1.0, color=group_color, linewidth=4, alpha=0.8)
                
                # Formatting
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Endo', 'NatBG'], fontsize=14)
                ax.set_ylabel(f'{score_name} Score', fontsize=14)
                
                # Title
                sig_marker = '*' if p_val < 0.05 else ''
                diff = float(row[diff_col])
                title_text = f"{row['config']}_seq{row['ce_sequence_index']}{sig_marker}\ndiff={diff:.3f} p={float(p_val):.3f}"
                ax.set_title(title_text, fontsize=12, pad=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                if i == 0:  # Add legend to first plot
                    ax.legend(fontsize=10)
            
            plt.suptitle(f'Top 20 {score_name} Context Effects (Highest Differences)', fontsize=18, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save as SVG only
            focused_svg = f'DeepSTARR_dm6_top20_{score_type}_context_effects.svg'
            plt.savefig(focused_svg, format='svg', bbox_inches='tight')
            
            print(f"Focused {score_name} plot saved as {focused_svg}")
            plt.show()
    
    # Create plots for both score types
    create_comparison_plot('dev', 'DeepSTARR_dm6_dev_analysis.png')
    create_comparison_plot('hk', 'DeepSTARR_dm6_hk_analysis.png')
    
    # Create focused plots with top significant results
    create_focused_plots(individual_df, summary_df)

def main():
    """Main pipeline function"""
    print("Starting Comprehensive CE Analysis Pipeline")
    print(f"Using {BACKGROUNDS_PER_CE_SEQUENCE} natural backgrounds per CE sequence")
    print(f"Random seed: {RANDOM_SEED}")
    print("="*60)
    
    # Step 1: Run enrichment analysis
    df = run_enrichment_analysis()
    
    # Step 2: Generate multiple backgrounds per sequence
    expanded_df = generate_multiple_backgrounds(df)
    
    # Step 3: Setup DeepSTARR
    model = setup_deepstarr()
    
    # Step 4: Score all sequences
    individual_df = score_sequences(expanded_df, model)
    
    # Step 5: Create summary statistics
    summary_df = create_summary_statistics(individual_df)
    
    # Step 6: Save results
    print("=== Saving Results ===")
    individual_df.to_csv('DeepSTARR_dm6_individual_scores.csv', index=False)
    summary_df.to_csv('DeepSTARR_dm6_summary_statistics.csv', index=False)
    
    with open('DeepSTARR_dm6_individual_scores.pkl', 'wb') as f:
        pickle.dump(individual_df, f, protocol=4)
    with open('DeepSTARR_dm6_summary_statistics.pkl', 'wb') as f:
        pickle.dump(summary_df, f, protocol=4)
    
    print("Results saved to DeepSTARR_dm6_*.csv and DeepSTARR_dm6_*.pkl files")
    
    # Step 7: Create plots
    create_plots(individual_df, summary_df)
    
    print("="*60)
    print("Comprehensive CE Analysis Pipeline Complete!")
    
    return individual_df, summary_df

if __name__ == "__main__":
    individual_df, summary_df = main() 