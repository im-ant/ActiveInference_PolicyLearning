# ActiveInference_PolicyLearning
Software used for the Active Inference Policy Learning paper

## Dependencies Note
All of the work related to this paper is done in MATLAB (https://www.mathworks.com/products/matlab.html) using the Statistical Parametric Mapping 12 (SPM12) software (https://www.fil.ion.ucl.ac.uk/spm/). Specifically, the simulations need the DEM toolbox in SPM12.


## File Descriptions
#### twoStep_multiDayTraining.m
Script used for training the agents for multiple days in the two-step maze task. Options to save the agent's _e_ matrices to a `.mat` file.


#### testTrials_eRanges_twoStepTask.m
Script used to test the post-training agents. Two types of testing are available:
- Testing the agents' performance (reward acquisition rate) over a period of time
- Testing the agents' performance before and after training (or, testing the performance of a naive agent against a trained one)


#### spm_MDP_VB_Xnew.m
The solver which implements a discrete version of Active Inference, the basis of the simulation which the above two script requires.

Note that this solver can be found under the DEM toolbox in SPM12. The one found in SPM12 should be referred to as the official solver. This is put here only to show the version we have used at this time.


#### get_bayesianAvg_reducedPriors.m
Function used to generate Bayesian model reduced and averaged _e_ parameters, based on prior and posterior full _e_ parameters. Reduced models are generated automatically via iterating over all combinations of which parameter to reduce, and Bayesian model averaging averages over the entire model space to generate the final model.
