# uiuc-cs598-dlh-project

# Reproducing Survival Cluster Analysis (SCA)

This repository contains code to reproduce experiments from the paper "Survival Cluster Analysis" by Chapfuwa et al. (2020), specifically focusing on the SCA model implementation for the SUPPORT dataset.

The implementation uses TensorFlow 1.x compatibility mode and was developed and tested primarily in a Google Colab environment. This README provides instructions for running the main SCA model reproduction script and includes notes on the debugging process encountered.

## Environment Setup (Google Colab)

This code is designed to be run in a Google Colab notebook environment.

1.  **Runtime:** Ensure your Colab runtime has GPU acceleration enabled ("Runtime" -> "Change runtime type" -> "Hardware accelerator" -> "T4 GPU" or similar).
2.  **Python:** The code was tested with Python 3.11 (typical Colab version).

## Dependencies Installation

Before running the main script, you need to install the required packages. Run the following commands in a cell in your Colab notebook:

```python
# Install main dependencies
# Using pandas 1.5.3 might help with Colab compatibility initially, adjust if needed
!pip install pandas==1.5.3 lifelines matplotlib seaborn scikit-learn tensorflow requests
```

# Note on Environment Stability

TensorFlow 1.x compatibility mode and library interactions can sometimes cause issues (such as NumPy dtype errors).  
If you encounter unexpected errors after installation or during execution:

- Restart the runtime using the Colab menu ("Runtime" → "Restart runtime...")
- Then run the script cell again without re-running the installation cell.

# Running the Code

1. **Upload/Open:** Upload the main Python script, open the `.ipynb` file.
2. **Install Dependencies:** Run the pip install cell provided above. Restart the runtime if necessary (see the note above).
3. **Execute Script:** Run the main cell containing the Python code for the SCA model.

The script will perform various steps including data loading and preprocessing, model building, training, evaluation, and saving results.

# Expected Output

Upon successful execution, you should expect to see:

- Print statements indicating data download, imputation details, and dataset sizes.
- Debugging print statements (e.g., `--- ... ---`) during model graph construction.
- Output for each training epoch showing Train Cost, Train CI, Valid Cost, Valid CI, Best Valid CI, and elapsed time.
- A message indicating training finished and a cost plot saved (`SCA_cost.png`).
- Evaluation results printed for the SCA model on the test set (Test Cost, Test CI, etc.; expect Test CI around 0.51, significantly lower than the paper).
- "TensorFlow session closed." and "Done!" messages.

# Notes on Reproducibility and Debugging

This section summarizes the key challenges encountered and solutions applied during the reproduction effort:

## Initial Code Base

The starting point was code likely derived from the original paper’s repository, aiming to implement the SCA model.

## TensorFlow Version Conflicts

Early errors (such as `ValueError: as_list()` and `AttributeError`) stemmed from conflicts between TensorFlow 1.x compatibility mode (`tf.compat.v1`) and the use of modern `tf.keras.layers`.  
Switching between `tf.compat.v1.layers` and `tf.keras.layers` revealed further incompatibilities depending on the TensorFlow/Keras backend version.

**Solution:** Implemented `tf.cond` within the network functions (`pz_given_x`, `pt_given_z`) to explicitly define separate graph execution paths for training (`training=True`) and inference (`training=False`).

## Numerical Instability (NaN/Inf)

The model was highly prone to producing NaN or Inf values during training, often detected by `tf.debugging.check_numerics` or causing errors in the Concordance Index (CI) calculation.

**Initial Cause (Data):**  
The initial use of `dropna()` on the SUPPORT dataset drastically reduced the sample size (~9,105 to ~400), likely contributing to instability.  
Replacing `dropna()` with **median/mode imputation** restored the dataset size and improved stability.

**Persistent Cause (Model Computation):**  
Even after fixing the data, Inf values occurred in the Weibull prediction calculation:  
`pow_result = tf.pow(pow_base, pow_exponent)`  
This was traced to the exponent `1.0 / alpha_clipped` becoming extremely large when predicted alpha values approached zero.

**Solutions:**

- Added extensive `tf.debugging.check_numerics` to pinpoint sources of Inf/NaN.
- Implemented **gradient clipping** (`tf.clip_by_value`) during optimization.
- Explicitly **clipped** the outputs of the `alpha` and `beta` layers (setting lower bounds like 1e-4).
- Clipped the final `predicted_time` output to avoid exploding values.
- Switched weight initialization from `GlorotNormal` to `RandomNormal(stddev=0.01)`.
- Reduced the learning rate to **1e-4**.

## Batch Normalization Updates

Debug logs showed `Found 0 update ops` when using `tf.compat.v1.get_collection`, indicating potential issues with BatchNorm updates.  
The explicit control dependencies (`tf.control_dependencies`) were removed, relying on TensorFlow/Keras optimizers to handle BatchNorm updates implicitly, which proved sufficient once other numerical issues were addressed.

# Final Performance

Despite achieving a stable implementation:

- The final **Test Concordance Index (CI)** achieved was approximately **0.51**,
- which is significantly lower than the paper's reported **0.85**.

This suggests that either:

- Crucial loss components (such as `l_dp`, `l_cal`) may be missing from the implemented objective function, or
- Further hyperparameter tuning is required.

