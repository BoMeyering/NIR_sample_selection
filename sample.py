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

from argparse import ArgumentParser

from src.utils import validate_args, filter_spectra, EquidistantInterpolator

parser = ArgumentParser()

parser.add_argument('input_file', type=str, help='The csv file which contains the raw spectra', default='measurements.csv')
parser.add_argument('-f', '--format', help="The format of the spectra. One of 'wavenumber' or 'wavelength'", default='wavelength')

args = parser.parse_args()

# Argument validation
validate_args(args)

def main(args: argparse.Namespace=args):
    """ Run spectral processing and sample selection algorithms on the input file """

    # Read in spectra
    raw_df = pl.read_csv(args.input_file)

    # Filter low quality spectra
    _, idx = filter_spectra(X=raw_df[:, 4:].to_numpy(), threshold=10)
    raw_df = raw_df.filter(idx)

    # Return new df with sample name and spectral columns
    filtered_spectra = raw_df.select([raw_df.columns[0]] + raw_df.columns[4:])
    print(filtered_spectra)

    # Grab wavelengths
    lambda_nm = np.array(list(map(lambda x: round(float(x), 2), filtered_spectra[:, 1:].columns)))
    print(type(lambda_nm))

    print(lambda_nm)
    eq_interp = EquidistantInterpolator(lambda_nm)
    filtered_spectra = filtered_spectra[:, 1:].to_numpy()
    X = eq_interp.fit_transform(filtered_spectra)

    return raw_df

if __name__ == '__main__':
    main(args)