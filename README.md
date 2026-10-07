# Wavelet Analysis: Coherence and Cross-Power

This repository contains two Jupyter notebooks for running continuous wavelet analysis on time series, including wavelet coherence and cross-wavelet power, with and without significance testing.

## Contents

| File | Description |
|------|-------------|
| `wavelet_analysis.ipynb` | Wavelet analysis including coherence and crosspower. |
| `wavelet_analysis_significance.ipynb` | The same analysis, with significance calculation. |
| `wavelet_wrapper_posh` | folder containing all module with all functions needed to run the analyses. |

## Background

The functions in `wavelet_wrapper_posh` are based on the methods and codebase of [O'Malley et al. (2023)](https://onlinelibrary.wiley.com/doi/full/10.1111/geb.13702) ([codebase](LINK_TO_OMALLEY_CODEBASE)). They build on the [`pycwt`](https://github.com/regeirk/pycwt) library and also implement rectified power, following [Liu et al. (2007)](https://journals.ametsoc.org/view/journals/atot/24/12/2007jtecho511_1.xml).

## Getting started
1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
   ```

2. Install the dependencies (adjust to match your environment), e.g.:

   ```bash
   pip install pycwt numpy pandas matplotlib dask 
   ```

3. Launch Jupyter, open and run one of the notebooks

## Usage

- Use **`wavelet_analysis.ipynb`** for a quick run of wavelet coherence and crosspower on your data.
- Use **`wavelet_analysis_significance.ipynb`** when you also need significance estimates for the results.

Both notebooks import their functions from the .py files in `wavelet_wrapper_posh`, so keep that file in the same directory as the notebooks.

## References

- O'Malley, et al. (2023). *Global Ecology and Biogeography*. https://onlinelibrary.wiley.com/doi/full/10.1111/geb.13702
- Liu, Y., et al. (2007). *Journal of Atmospheric and Oceanic Technology*, 24(12). https://journals.ametsoc.org/view/journals/atot/24/12/2007jtecho511_1.xml

