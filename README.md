# Statistical Analysis of Respiratory Data for Differentiating Patients with Wheezing and crackles.

This project analyzes a respiratory dataset to determine if there are statistically significant differences between patients with wheezing and healthy individuals. The workflow includes:

## Data Preprocessing:

- Filtering: Initial filtering is performed using high-pass and low-pass filters to clean the data.

- Wavelet Decomposition: Wavelet decomposition is applied to remove cardiac rhythm artifacts from the respiratory signals.

## Statistical Analysis:

- Assumptions Check: Statistical assumptions are verified to prepare for parametric testing.

- Parametric Testing: A parametric test is conducted, but the assumption of homogeneity of variances (homoscedasticity) is not met.

- Non-Parametric Testing: Given the failure to meet homoscedasticity, a non-parametric U-test is performed.

## Results: 

The analysis reveals significant differences between each combination of patient groups in all cases.

## Key Findings:

- Significant statistical differences were observed between patients with wheezing and healthy individuals.

- Non-parametric methods provided conclusive results despite the failure of parametric assumptions.