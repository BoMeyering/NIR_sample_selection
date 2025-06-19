# NIR_sample_selection
Process raw NIR spectra and select a subset of samples for wet chemistry analysis

## What this is used for
Takes a dataframe of sample names and spectra from NeoSpectra portal and selects a subset of the samples for wet chemistry analysis
You can run the script using
```
$ python sample.py <input_file.csv> -f 'wavelength' -s 0.15
```
The script accepts one positional argrument 'input-file' which only accepts .csv formatted tables (X)
'--format/-f' is the format that the values are in. Must be one of 'wavelength' or 'wavenumber', Case-insensitive
'--sample-size' is the proportion of samples you want to select. Must be a float in the interval [0, 1] or an integer < nrow(X).

This script runs a pipeline that linearly interpolates the data across the spectrum, asserting equidistant data points, fits a Savitsky-Golay filter with a 2nd derivative, and then performs mean-centering standard scaling before runnning the Kennard-Stone selection algorithm for the selected sample size.

The script outputs a .csv file with the selected sample names.

## To Come
The argument will have optional parameters for the window-length and polyorder for Savitsky-golay filter and the ability to convert wavenumber formatted data to wavelength before sampling
