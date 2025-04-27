# A substellar flyby that shaped the orbits of the giant planets: code
A repository of code for scientific article https://arxiv.org/abs/2412.04583.
The data for this article can be found at https://doi.org/10.5281/zenodo.15288677 and the *vanilla* solar system data used as part of the analysis can be found at https://doi.org/10.5281/zenodo.4299102.

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

Here is a brief description of each of the files in the data directory (hosted on Zenodo).

- `flyby_19399_details.sa`: a REBOUND Simulationarchive with detailed sampling for generating the perihelion figure in the paper.
- `flyby_metric.npy`: the results of comparing the frequency decomposition of each flyby system to every solar system decomposition using the log spectral distance metric, saved as a numpy array.
- `flyby_secular_modes.secs`: the frequency decompositions of each flyby system as a Results object defined by the companion pal.secular code, saved as a pickled list of pal.secular.Results objects.
- `flyby.stars`: the flyby parameters for all the stars in the flyby simulations, saved as an AIRBALL.Stars object.
- `flybys_amd.npy`: the angular momentum deficit (AMD) at the end of the flyby simulations, saved as a numpy array.
- `initial_conditions.sims`: the initial conditions of all the systems *before* experiencing a flyby event, saved as a pickled list of REBOUND.Simulations.
- `oc.se`: the details of the open cluster environment that the stars were sampled from, saved as an AIRBALL.StellarEnvironment object.
- `post_20Myr_flyby_conditions.sims`: the final state of all the flyby systems after the additional 20 Myrs of integration (or less depending on whether or not a collision or escape event occurred), saved as a pickled list of REBOUND.Simulations.
- `post_flyby_conditions.sims`: the state of all the flyby systems *immediately* after the removal of the flyby star, saved as a pickled list of REBOUND.Simulations.
- `solar_system_20Myr.sec`: the frequency decomposition of a solar system integration, from the *vanilla* solar system set, saved as a pal.secular.Results object.
- `solar_system_metric.npy`: the results of comparing the frequency decomposition of a random subset of 2500 solar system to additionally random subsets of 2500 solar system decompositions using the log spectral distance metric, saved as a numpy array.
- `solar_system_secular_modes.secs`: the secular frequency decompositions of all of the 20 Myr slices of the *vanilla* solar system dataset, saved as a pickled list of pal.secular.Results objects.