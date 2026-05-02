# COMSOL Photoacoustic FFT Analyzer

A small Python/Streamlit tool for FFT-based frequency-domain analysis of COMSOL-exported time-domain pressure signals.

The tool is designed for post-processing photoacoustic simulation outputs. It reads COMSOL CSV exports, separates optional parametric sweep cases, applies baseline correction and Hann windowing, computes normalized FFT spectra, and reports dominant frequency and half-maximum bandwidth (BW50).

## Features

- Upload COMSOL CSV files in a browser interface
- Command-line interface for single-file or batch analysis
- Optional parametric sweep support, such as pulse duration or other model parameters
- Baseline correction using a user-defined pre-pulse interval
- Hann-windowed real FFT
- Normalized spectrum plots
- Dominant frequency extraction
- Half-maximum bandwidth (BW50) calculation
- PNG, CSV, and TXT report outputs

## Repository contents

```text
app.py                     # Streamlit web interface
fft_core.py                # Core parsing, FFT, plotting, and export functions
cli.py                     # Command-line interface
batch_config_example.json  # Example batch-mode configuration
requirements.txt           # Python dependencies
examples/                  # Synthetic example input data
outputs/                   # Generated outputs, ignored by git
```

## Installation

Clone or download this repository, then install the dependencies:

```bash
py -m pip install -r requirements.txt
```

On some systems, use `python` instead of `py`:

```bash
python -m pip install -r requirements.txt
```

## Run the Streamlit app

```bash
py -m streamlit run app.py
```

Then open the local URL shown in the terminal. The app lets you upload a CSV file, select the time/signal/parameter columns, adjust FFT settings, run the analysis, and download the results.

## Run from the command line

Single-file analysis:

```bash
py cli.py single --csv examples/sample_comsol_pressure.csv --point-name "Sample Point" --param-column "tau_p (ns)" --time-column "Time (ns)" --signal-column "Synthetic pressure (Pa)" --output-dir outputs
```

Batch analysis:

```bash
py cli.py batch --config batch_config_example.json
```

## Expected COMSOL CSV format

The tool supports COMSOL-style CSV files with metadata rows beginning with `%`. The last metadata row may be used as the column header.

Example:

```text
% Model, Synthetic COMSOL-like example
% tau_p (ns),Time (ns),Synthetic pressure (Pa)
5,0.0,1.23e-10
5,0.2,2.34e-10
10,0.0,9.87e-11
10,0.2,1.11e-10
```

A parameter column is optional. If there is no parameter column, the full file is treated as one signal.

## FFT methodology

1. The mean value over the pre-pulse interval is subtracted as a baseline correction.
2. A Hann window is applied to reduce spectral leakage.
3. The real FFT is computed with `numpy.fft.rfft`.
4. The frequency axis is converted to MHz.
5. Spectra are normalized by the maximum non-DC amplitude.
6. The dominant frequency is identified after excluding the DC component.
7. BW50 is calculated from the half-maximum crossings around the dominant peak using linear interpolation.

## Notes for private simulation data

Do not upload private COMSOL files, thesis drafts, or raw research data unless you intend them to be public. The `.gitignore` file excludes typical generated outputs and COMSOL/private data extensions. The included CSV file in `examples/` is synthetic demonstration data only.

## License

This project is released under the MIT License.
