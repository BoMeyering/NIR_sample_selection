"""
sample.py
NIR Sample Selection Script
BoMeyering 2025
"""

# Module imports
import scipy
import sklearn
import argparse
import numpy as np
import polars as pl
import astartes as at
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from argparse import ArgumentParser

from src.utils import validate_args, filter_spectra, EquidistantInterpolator, SavgolTransform

parser = ArgumentParser()

parser.add_argument('input_file', type=str, help='The csv file which contains the raw spectra', default='measurements.csv')
parser.add_argument('-f', '--format', help="The format of the spectra. One of 'wavenumber' or 'wavelength'", default='wavelength')
parser.add_argument('-s', '--sample-size', type=float, help='The proportion or absolute number of samples you want to return', default=0.15)

# Parse the arguments
args = parser.parse_args()

# Argument validation
validate_args(args)

def main(args: argparse.Namespace=args):
    """ Run spectral processing and sample selection algorithms on the input file """

    # Read in spectra
    raw_df = pl.read_csv(args.input_file)

    # Check format and convert to wavelengths
    if args.format.lower() != 'wavelength':
        pass

    # Filter low quality spectra
    _, idx = filter_spectra(X=raw_df[:, 4:].to_numpy(), threshold=10)
    raw_df = raw_df.filter(idx)

    # Return new df with sample name and spectral columns
    sample_names = raw_df.select([raw_df.columns[0]])
    raw_spectra = raw_df.select(raw_df.columns[4:]).to_numpy()


    # Grab wavelengths
    lambda_nm = np.array(list(map(lambda x: round(float(x), 2), raw_df[:, 4:].columns)))

    pipe = Pipeline(
        [
            ('interp', EquidistantInterpolator()),
            ('savgol', SavgolTransform()),
            ('scaler', StandardScaler())
        ]
    )

    # Set pipeline parameters
    pipe.set_params(
        interp__lambda_nm=np.array(lambda_nm),
        savgol__window_length=13,
        savgol__polyorder=3,
        savgol__deriv=2,
        scaler__with_std=False
    )

    # Fit the pipeline
    out = pipe.fit_transform(raw_spectra)

    sample_names = sample_names.to_numpy()

    # Convert absolute samples to proportion
    if args.sample_size > 1:
        nrow = out.shape[0]
        args.sample_size = args.sample_size / nrow

    X_train, X_test, names_train, names_test = at.train_test_split(out, sample_names, train_size=args.sample_size)

    return names_train

if __name__ == '__main__':
    sample_names = main(args)

    pl.DataFrame({'sample_name': sample_names.flatten()}).write_csv(f'sample_selections_{args.sample_size}.csv')