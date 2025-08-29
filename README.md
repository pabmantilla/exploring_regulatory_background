# Exploring the Regulatory Background

Pipeline for scoring Composite Enhancers (CEs) in Drosphila (dm6) with DeepSTARR then analyzing CE-wise and background-wise.

Each script default returns a df. Check docuementation of each to see all the plotting options, alternativley use just the Plot scripts to plot. Another option is to only create df with Create scripts and use Experiements scripts to run each experiment. When trying a new motif file, it's recommended that you choose sanity_check_plot='True'. Check doc for details. 

Sections:
  1) Create
  2) Experiments
  3) Utils.py

  
- **Create**
  - `search.py (GPU)` → CE search & scoring  
  - `quality_control.py` → QC & filtering  
  - `sanity_plot.py` → Sanity check plots
    - CE_endo_nat_bg
        

- **Experiments**
  - `check_Exp_X.py` → Verify inputs & perturbations (e.g. TF-MoDISco)  
  - `run_Exp_X.py` → Run experiment scoring (DeepSTARR Dev & Hk)
  - `plot_Exp_X.py` → Generate experiment plots 

- **Utils**
  - `utils.py` → Shared helper functions  


