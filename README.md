# Secular-Origin
A repository of code for scientific article https://arxiv.org/abs/2412.04583.
The data for this article can be found at https://doi.org/10.5281/zenodo.15288677 and the solar system data used as part of the analysis can be found at https://doi.org/10.5281/zenodo.4299102.

## Installation

The source code can be downloaded with
```sh
git clone https://github.com/zyrxvo/Secular-Origin.git
```
and the dependencies of this project can be installed with
```sh
pip install -r requirements.txt
pip install .
```
or using [uv](https://docs.astral.sh/uv/) and `uv sync`.

The data generated for this article can uncompressed into the `data/` directory.
It can be downloaded with
```sh
wget https://zenodo.org/records/15288677/files/data.zip
```
or by visiting the link.

## Data

The main flyby data was generated in parallel using
```sh
python opencluster.py 50000 <index>
```
for each `index` up to 50,000.

After simulating the flybys for each initial condition, each simulation was simulated for an additional 20 Myr (saving 2048 samples), or until a collision or escape.
Then, each `Simulationarchive` was analyzed for the complex eccentricities and inclinations to convert into frequency space for use with our log spectral distance metric.
The analysis is done in the `secular-origin.ipynb` notebook.

### Data files

Here is a brief description of each of the files in the data directory.

- `flyby_19399_details.sa`: a REBOUND Simulationarchive with detailed sampling for generating the perihelion figure in the paper.
- `flyby_metric.npy`: 