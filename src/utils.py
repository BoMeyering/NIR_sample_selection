"""
src/utils.py
Utility Functions
BoMeyering 2025
"""

import os
import argparse
import numpy as np
import polars as pl
import astartes as at
import matplotlib.pyplot as plt
from typing import Iterable
from itertools import compress
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

# Filter function to get rid of low quality spectra
def filter_spectra(X: np.ndarray, threshold: int) -> np.ndarray:
    """
    Takes a Numpy array and filters out any spectra with low spread between the minimum and maximum value of the spectra
    Returns a filtered Numpy array of spectra
    """
    spread = X.max(axis=1) - X.min(axis=1)
    idx = spread > threshold

    return X[idx], idx

def wavenumber_to_nm(X: Iterable):
    """ Convert Iterable wavenumber per cm to wavelength(nm)"""
    # Sort X from hi to lo
    X = sorted(list(X), reverse=True)
    # Convert from wavenumber/cm to wavelength(nm)
    nm = list(map(lambda x: 10**7 / float(x), X))
    
    return nm
    
def nm_to_wavenumber(X):
    """ Convert Iterable wavelength(nm) to wavenumber per cm """
    # Sort X from lo to hi
    X = sorted(list(X))
    # Convert from wavelength(nm) to wavenumber/cm
    wavenumber = list(map(lambda x: 1/(10**-7 * float(x)), X))

    return wavenumber

def error_fig(gt, preds, title, block=None):
    mapping = {
        'ward': 'red',
        'dairyland': 'blue'
    }
    fig, ax = plt.subplots()
    if block is not None:
        for block_name in np.unique(block):
            idx = [x == block_name for x in block]
            sub_preds = list(compress(preds, idx))
            sub_gt = list(compress(gt, idx))
            ax.scatter(sub_gt, sub_preds, color=mapping.get(block_name), alpha=0.6, label=block_name)
        ax.legend(loc="upper left", title="Wetchem origin")
    else:
        ax.scatter(gt, preds, color='blue', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel("Predictions")
    ax.axline((0, 0), slope=1, color='green', linestyle='--')

    min_val = min(gt.min(), preds.min())
    max_val = max(gt.max(), preds.max())

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    plt.close(fig)

    return fig

class EquidistantInterpolator(BaseEstimator):
    """
    Ascending sort and linearly interpolate data with equidistant wavelength steps
    """

    def __init__(self, lambda_nm: np.ndarray=None):
        self.lambda_nm = lambda_nm

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        
        # Sort the wavelength vector
        sort_idx = np.argsort(self.lambda_nm).astype(int)
        lambda_nm = self.lambda_nm[sort_idx]
        min_nm = np.min(lambda_nm) # Grab the min and max wavelength
        max_nm = np.max(lambda_nm)
        # Set up an equidistant sampling space
        equi_nm = np.linspace(start=min_nm, stop=max_nm, num=len(lambda_nm))
        self.equidistant_lambda_nm = equi_nm
        X = X[:, sort_idx]
        
        # Reinterpolate the array
        self.transformed_X = np.array(
            [
                np.interp(
                    x=equi_nm,
                    xp=lambda_nm,
                    fp=row
                ) for row in X
            ]
        )

        return self.transformed_X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
        

class SavgolTransform(BaseEstimator):
    """
    Implement a Savitsky Golay filter as a preprocessing step
    Returns the filtered, derivitized data
    """    
    def __init__(self, window_length: int=10, polyorder: int=5, deriv: int=2):
        """ Initialize and allow keyword arguments to be passed """
        self.window_length=window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X=None, y=None):
        """ Not implemented, return self """
        return self

    def transform(self, X):
        """ Transform the X matrix of spectra with a Savgol filter according to initialized attributes """
        X_filtered = savgol_filter(
            x=X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv
        )
        return X_filtered

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def validate_args(args: argparse.Namespace):
    """_summary_

    Args:
        args (argparse.Namespace): _description_

    Returns:
        _type_: _description_
    """
    # Check for required values
    if 'input_file' not in args:
        raise ValueError(f"Positional argument 'input_file' is missing.")
    elif not os.path.exists(args.input_file):
        raise ValueError(f"Path to the input file {args.input_file} does not exist. Please ensure you passed the correct path.")
    elif not os.path.basename(args.input_file).endswith('.csv'):
        raise ValueError(f"Input file '{args.input_file}' must be a .csv file.")
    
    # 'format'
    if 'format' not in args:
        raise ValueError()
    elif args.format.lower() not in ['wavelength', 'wavenumber']:
        raise ValueError()

    # 'sample-size'
    if 'sample_size' not in args:
        raise ValueError("A numeric value must be passed to '--sample-size'.")
    elif not isinstance(args.sample_size, (float, int)):
        raise ValueError(f"'--sample-size' must be either a 'float' or an 'int'.")
    elif args.sample_size < 0:
        raise ValueError("'--sample-size' must be greater than 0.")
    
    return True