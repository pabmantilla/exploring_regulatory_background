# exploring_regulatory_background

Pipeline for scoring Composite Enhancers (CEs) in Drosphila (dm6) with DeepSTARR then analyzing CE-wise and background-wise.

Each script default returns a df. Check docuementation of each to see all the plotting options, alternativley use just the Plot scripts to plot. Another option is to only create df with Create scripts and use Experiements scripts to run each experiment. When trying a new motif file, it's recommended that you choose sanity_check_plot='True'. Check doc for details. 

Sections:
  1) Create (search.py(gpu), quality_control.py, sanity_plot.py)
  2) Experiments (exp_X_check.py, run_Exp_X.py, plot_Exp.py)
  3) Utils.py


#Create


Experiments 
  exp_X_check.py
    Checks that all information is present. Verifies that all perturbations (tfmodisco).
  run_Exp_X.py
    Runs the experiment (scoring).
  plot_Exp.py
    Plots results.


    
#Utils.py



