import rebound
import airball
import airball.units as u
from copy import deepcopy
import json
import pickle
import warnings
import numpy as np
from celmech.miscellaneous import frequency_modified_fourier_transform as fmftc

twopi = 2 * np.pi


def logSpectralDistance(P1, P2, outer=True):
    """
    The log spectral distance [LSD](https://en.wikipedia.org/wiki/Log-spectral_distance)

    This function assumes that the input are numpy arrays of the same shape.
    The outer parameter is used to only compare the outer planets if the system has more than 4 planets.
    """
    tot = 0
    n1 = len(P1)
    n2 = len(P2)
    range1 = [i for i in range(n1)]
    range2 = [i for i in range(n2)]
    if n1 > 4 and outer:
        range1 = [i for i in range(4, n1)]
    if n2 > 4 and outer:
        range2 = [i for i in range(4, n2)]

    if len(range1) != len(range2):
        raise Exception("Comparison of different number of planets.")

    for i1, i2 in zip(range1, range2):
        tot += np.sqrt(
            np.mean((np.log10(np.abs(P1[i1])) - np.log10(np.abs(P2[i2]))) ** 2)
        )
    return tot / len(range1)


def metric(SOLAR_SYSTEM, system, outer=True):
    """Using the LSD to define a metric using the eccentricity and inclination secular modes."""
    if not system.good or np.any(np.isnan(system.zfft)):
        return np.inf
    total = logSpectralDistance(SOLAR_SYSTEM.zfft, system.zfft, outer)
    total += logSpectralDistance(SOLAR_SYSTEM.ζfft, system.ζfft, outer)
    return total / 2


def solar_subset(sec, ssdat, size=2500):
    """Comparing subsets of the secular Solar System data against itself using the metric."""
    if not sec.good:
        return [np.inf] * size
    lset = []
    for s in np.random.choice(ssdat, size=size, replace=False):
        lset.append(metric(s, sec))
    return lset


def solar(sec, ssdat):
    """Comparing a secular system too *all* of the secular Solar System data."""
    if not sec.good:
        return np.inf * np.ones(len(ssdat))
    lsol = np.zeros(len(ssdat))
    for i, s in enumerate(ssdat):
        lsol[i] = metric(s, sec)
    return lsol


def pkl(filename):
    """Save a pickle file."""
    with open(filename, "wb") as pfile:
        pickle.dump(filename, pfile, protocol=pickle.HIGHEST_PROTOCOL)


def unpkl(filename):
    """Load a pickle file."""
    with open(filename, "rb") as pfile:
        return pickle.load(pfile)


def extract(filename):
    """Returns n, t, z, ζ, a from a rebound simulation archive file."""
    sa = rebound.Simulationarchive(filename)
    N = sa.nblobs
    n = sa[0].N - 1
    t = np.zeros(N)
    a = np.zeros((N, n))
    z = np.zeros((N, n), dtype=complex)
    ζ = np.zeros((N, n), dtype=complex)
    for i in range(N):
        s = sa[i]
        s.synchronize()
        t[i] = s.t
        o = s.orbits()
        for j in range(n):
            a[i, j] = o[j].a
            z[i, j] = o[j].e * np.exp(1j * o[j].pomega)
            ζ[i, j] = np.sin(o[j].inc / 2.0) * np.exp(1j * o[j].Omega)
    return n, t, z, ζ, a


def fmft(filename, Nfreq, method_flag=3):
    """
    Returns the Fourier modes of the z and ζ variables from a rebound simulation archive file.

    This function uses the frequency modified Fourier transform (FMFT) written by David Nesvorny and accessed through the celmech package.
    """
    n, t, z, ζ, _ = extract(filename)
    g = []
    s = []
    for i in range(n):
        g.append(fmftc(t, z[:, i], Nfreq, method_flag=method_flag))
        s.append(fmftc(t, ζ[:, i], Nfreq, method_flag=method_flag))
    return g, s


def plist(arr, end=""):
    print("\n".join([f"{k:10.8f}" for k in arr]) + end)


def secularmodes(fftdecomp, atol=1e-8):
    """
    Returns the secular modes and amplitudes from the Fourier decomposition of the z or ζ variables.

    This function assumes that the input is a list of dictionaries, where each dictionary contains the Fourier decomposition of a single planet.
    The keys of the dictionary are the frequencies and the values are the complex amplitudes.
    The function returns the secular modes and amplitudes as numpy arrays.
    The function also assumes that the first frequency in the Fourier decomposition is the fundamental frequency of the planet.
    The function uses the absolute tolerance to determine if two frequencies are close enough to be considered the same mode.
    """
    n = len(fftdecomp)
    nf = len(fftdecomp[0])
    f = np.zeros((n, nf))
    A = np.zeros((n, nf))
    for i in range(n):
        for k, key in enumerate(fftdecomp[i]):
            f[i, k] = key
            A[i, k] = np.abs(fftdecomp[i][key])

    # Assume f_i is the first frequency in fftdecomp_i
    modes = f[:, 0]
    amps = A[:, 0]
    D = np.abs((modes[:, np.newaxis] - modes))
    D[np.diag_indices_from(D)] = np.inf

    for i in range(n):
        inds = D[i] < atol
        j = 0
        while np.any(inds):
            j += 1
            if j >= nf:
                break
            if np.any(amps[i] < amps[inds]):
                modes[i] = f[i, j]
                amps[i] = A[i, j]
            else:
                break

            D = np.abs((modes[:, np.newaxis] - modes))
            D[np.diag_indices_from(D)] = np.inf
            inds = D[i] < atol

    return modes, amps


def Smatrix(fftdecomp, rtol=1e-5):
    """
    Returns the secular modes, amplitudes, and matrix from the Fourier decomposition of the z or ζ variables.

    Similar to the secularmodes function, but also returns the matrix S which contains how the amplitudes of the different modes are related to each other.
    """
    n = len(fftdecomp)
    nf = np.max([len(fftdecomp[i]) for i in range(n)])
    f = np.zeros((n, nf))
    A = np.zeros((n, nf), dtype=complex)
    for i in range(n):
        for k, key in enumerate(fftdecomp[i]):
            f[i, k] = key
            A[i, k] = fftdecomp[i][key]
    # Assume f_i is the first frequency in fftdecomp_i
    modes = f[:, 0].copy()
    amps = A[:, 0].copy()

    # if mode is 0, Catch divide by zero warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D = np.abs((modes[:, np.newaxis] - modes) / modes)
    D[np.diag_indices_from(D)] = np.inf

    for i in range(n):
        inds = D[i] < rtol
        j = 0
        while np.any(inds):
            j += 1
            if j >= nf:
                break
            if np.any(np.abs(amps[i]) < np.abs(amps[inds])):
                modes[i] = f[i, j].copy()
                amps[i] = A[i, j].copy()
            else:
                break

            # if mode is 0, Catch divide by zero warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                D = np.abs((modes[:, np.newaxis] - modes) / modes)
            D[np.diag_indices_from(D)] = np.inf
            inds = D[i] < rtol

    S = np.zeros((n, n), dtype=complex)
    for i in range(n):  # Planet
        for j in range(n):  # Mode
            a = 0
            for k in range(nf):  # Frequency
                # if mode is 0, Catch divide by zero warning.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rdiff = np.abs((f[i, k] - modes[j]) / modes[j])
                if rdiff < rtol:
                    a = A[i, k]
            S[i, j] = a

    return modes, amps, S


def match(modes, target, atol=1e-8):
    rdiffs = np.zeros(modes.shape[0])
    for i, m in enumerate(modes):
        rdiffs[i] = np.abs(target - m)
    ii = np.argmin(rdiffs)
    if rdiffs[ii] <= atol:
        return ii
    else:
        return None


def nAMD(sim, plane="invariable"):
    """
    Returns the normalized angular momentum deficit (nAMD) of the system as a numerator and denominator.
    The nAMD is defined as the difference between the angular momentum of the system and the angular momentum of a circular orbit with the same semi-major axis and mass.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = sim.copy()
        us = airball.tools.rebound_units(sim)
        airball.tools.rotate_into_plane(s, plane)
        s.move_to_com()
        num = np.zeros(s.N - 1)
        den = np.zeros(s.N - 1)
        mu = airball.tools.gravitational_mu(sim, star_mass=0)
        for i, p in enumerate(s.particles[1:]):
            try:
                num[i] = (
                    (p.m * us.mass * np.sqrt(mu * p.a * us.length))
                    * (
                        1
                        - np.cos(p.inc * u.rad)
                        * np.sqrt(1 - (p.e * u.dimensionless_unscaled) ** 2.0)
                    )
                ).value
                den[i] = (p.m * us.mass * np.sqrt(mu * p.a * us.length)).value
            except:
                num[i] = np.nan
                den[i] = np.nan
        del s
        return num, den


def divide_SA_into_Results(filename, index=None, power=2048):
    """Divide a simulation archive into Results objects."""
    sa = rebound.Simulationarchive(filename)
    N = len(sa)
    sim = sa[0]
    n = sim.N - 1

    t = np.zeros(N)
    a = np.zeros((N, n))
    z = np.zeros((N, n), dtype=complex)
    ζ = np.zeros((N, n), dtype=complex)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N):
            sim = sa[i]
            sim.synchronize()
            airball.tools.rotate_into_plane(sim)
            t[i] = sim.t
            orbits = sim.orbits()
            for j in range(n):
                a[i, j] = orbits[j].a
                z[i, j] = orbits[j].e * np.exp(1j * orbits[j].pomega)
                ζ[i, j] = np.sin(orbits[j].inc / 2.0) * np.exp(1j * orbits[j].Omega)

    sets = int(N // power)
    res = []
    for i in range(sets):
        r = Results()
        r.N = power
        r.n = n
        r.Nfreq = 2 * n
        r.t = t[i * power : (i + 1) * power].copy()
        r.a = np.median(a[i * power : (i + 1) * power], axis=0).copy()
        r.z = z[i * power : (i + 1) * power].copy()
        r.ζ = ζ[i * power : (i + 1) * power].copy()
        r.zfft = np.fft.fft(r.z, axis=0).T.copy()
        r.ζfft = np.fft.fft(r.ζ, axis=0).T.copy()
        r.freqs = np.fft.fftfreq(r.N, r.t[1] - r.t[0]) * twopi * 360.0 * 3600.0
        r.modes()
        if index is not None:
            r.index = (index, i)
        res.append(r)

    return res


class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder which turns numpy arrays into lists and complex values into dicts as these are not JSON serializable."""

    def default(self, obj):
        """The function called when an object is being serialized."""
        return self.replace_complex(obj)

    # turn complex to str
    def replace_complex(self, value):
        """Replace complex numbers with dicts and numpy arrays with lists.

        Adapted from: https://stackoverflow.com/a/56445571

        Parameters:
        ------------
        value: input dictionary.
        """

        try:
            if isinstance(value, complex):
                return {"re": value.real, "im": value.imag}
            elif isinstance(value, (list, np.ndarray)):
                if isinstance(value, u.Quantity):
                    value = value.value.tolist()
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                for each in range(len(value)):
                    value[each] = self.replace_complex(value[each])
                return value
            elif isinstance(value, dict):
                for key, val in value.items():
                    value[key] = self.replace_complex(val)
                return value
            else:
                return value  # nothing found - better than no checks
        except Exception as e:
            print(e)
            return ""


class CustomJSONDecoder(json.JSONDecoder):
    """JSON decoder which turns numpy arrays into lists as numpy arrays are not JSON serializable."""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.restore_complex, *args, **kwargs)

    def restore_complex(self, value):
        """Restore complex numbers and numpy arrays.

        Parameters:
        ------------
        value: input dictionary.
        """

        try:
            if isinstance(value, dict):
                if set(value.keys()) == {"re", "im"}:
                    return complex(value["re"], value["im"])
                for key, val in value.items():
                    value[key] = self.restore_complex(val)
                try:
                    value = {float(i): j for i, j in value.items()}
                except ValueError:
                    pass
                return value
            elif isinstance(value, list):
                for each in range(len(value)):
                    value[each] = self.restore_complex(value[each])
                return np.array(value)
            else:
                return value  # nothing found - better than no checks
        except Exception as e:
            print(e)
            return ""


class Results:
    """Class to store the results of a simulation.

    This class is used to store the results of a REBOUND.Simulationarchive in a way that can be easily saved and loaded.
    The class uses the pickle or json module to save and load the results.
    The class also uses the CustomJSONEncoder and CustomJSONDecoder classes to handle complex numbers and numpy arrays.
    """

    def __init__(self, load=None):
        self.good = True
        if load is not None:
            if isinstance(load, str):
                self.__dict__ = Results._load(load)
            elif isinstance(load, Results):
                self.__dict__ = deepcopy(load.__dict__)

    def save(self, filename):
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string.")
        if filename.endswith(".json"):
            with open(filename, "w") as pfile:
                json.dump(self.__dict__, pfile, cls=CustomJSONEncoder)
        else:
            with open(filename, "wb") as pfile:
                pickle.dump(self, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _load(cls, filename):
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string.")
        if filename.endswith(".json"):
            with open(filename, "r") as pfile:
                return json.load(pfile, cls=CustomJSONDecoder)
        else:
            with open(filename, "rb") as pfile:
                return pickle.load(pfile).__dict__

    def stats(self, returned=False):
        """
        Prints a summary of the current stats of the Results object.
        The stats are returned as a string if `returned=True`.
        """
        s = f"<{self.__module__}.{type(self).__name__} object at {hex(id(self))}, "
        s += f"good={self.good}"
        s += ">"
        if returned:
            return s
        else:
            print(s)

    def __str__(self):
        return self.stats(returned=True)

    def __repr__(self):
        return self.stats(returned=True)

    def secular_modes(
        self, SAfilename, Nfreq=None, power=2048, offset=1, fmft=True, warn="default"
    ):
        """Compute the secular modes of a simulation archive.

        The function uses the frequency modified Fourier transform (FMFT) written by David Nesvorny and accessed through the celmech package.
        The function also uses the Smatrix function to compute the secular modes and amplitudes.
        The offset parameter is used to skip the first few simulations in the archive. This is useful if the first simulation is the initial conditions of the system, before the stellar flyby.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(warn)
            sa = rebound.Simulationarchive(SAfilename)
            self.N = len(sa) - offset
            self.good = self.N >= power
            if self.good:
                sim = sa[0]
                self.n = sim.N - 1
                self.Nfreq = 2 * self.n if Nfreq is None else Nfreq

                self.t = np.zeros(power)
                a = np.zeros((power, self.n))
                self.z = np.zeros((power, self.n), dtype=complex)
                self.ζ = np.zeros((power, self.n), dtype=complex)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for i in range(power):
                        sim = sa[i + offset]
                        sim.synchronize()
                        airball.tools.rotate_into_plane(sim)
                        self.t[i] = sim.t
                        orbits = sim.orbits()
                        for j in range(self.n):
                            a[i, j] = orbits[j].a
                            self.z[i, j] = orbits[j].e * np.exp(1j * orbits[j].pomega)
                            self.ζ[i, j] = np.sin(orbits[j].inc / 2.0) * np.exp(
                                1j * orbits[j].Omega
                            )

                self.a = np.median(a, axis=0)
                if fmft:
                    self.gs = []
                    self.ss = []
                    for i in range(self.n):
                        self.gs.append(
                            fmftc(self.t - self.t[0], self.z[:, i], self.Nfreq)
                        )
                        if np.any(self.ζ[:, i]):
                            self.ss.append(
                                fmftc(self.t - self.t[0], self.ζ[:, i], self.Nfreq)
                            )
                        else:
                            self.ss.append(
                                {0: 1j * 0.0}
                            )  # if no inclination, set frequency analysis to 0.

                    self.g, self.A, self.Sg = Smatrix(self.gs)
                    self.g = (self.g << u.rad / u.yr2pi) << u.arcsec / u.yr

                    self.s, self.B, self.Ss = Smatrix(self.ss)
                    self.s = (self.s << u.rad / u.yr2pi) << u.arcsec / u.yr

                self.zfft = np.zeros((self.n, power), dtype=complex)
                self.ζfft = np.zeros((self.n, power), dtype=complex)
                self.freqs = (
                    np.fft.fftfreq(power, self.t[1] - self.t[0])
                    * twopi
                    * 360.0
                    * 3600.0
                )
                for k in range(self.n):
                    self.zfft[k] = np.fft.fft(self.z[:, k])
                    self.ζfft[k] = np.fft.fft(self.ζ[:, k])

    def modes(self, Nfreq=None):
        if self.good:
            if Nfreq is not None:
                self.Nfreq = Nfreq
            self.gs = []
            self.ss = []
            for i in range(self.n):
                self.gs.append(fmftc(self.t - self.t[0], self.z[:, i], self.Nfreq))
                if np.any(self.ζ[:, i]):
                    self.ss.append(fmftc(self.t - self.t[0], self.ζ[:, i], self.Nfreq))
                else:
                    self.ss.append(
                        {0: 1j * 0.0}
                    )  # if no inclination, set frequency analysis to 0.

            self.g, self.A, self.Sg = Smatrix(self.gs)
            self.g = (self.g << u.rad / u.yr2pi) << u.arcsec / u.yr

            self.s, self.B, self.Ss = Smatrix(self.ss)
            self.s = (self.s << u.rad / u.yr2pi) << u.arcsec / u.yr


class SecularResults:
    """Legacy class for the Results object."""

    def __init__(self, load=None):
        self.good = True
        if load is not None:
            if isinstance(load, str):
                loaded = SecularResults._load(load)
                self.__dict__ = loaded.__dict__
            elif isinstance(load, rebound.Simulation):
                self.sim = load
            else:
                self.__dict__ = load.__dict__
        else:
            self.sim = None
        if self.sim is not None:
            for p in self.sim.particles[1:]:
                self.good = self.good and p.a > 0

    def save(self, filename):
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string.")
        with open(filename, "wb") as pfile:
            pickle.dump(self, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _load(cls, filename):
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string.")
        with open(filename, "rb") as pfile:
            return pickle.load(pfile)

    def secular_modes(
        self,
        sim=None,
        tmax=20469961.575564947 * twopi,
        Nfreq=None,
        port=None,
        max_sample_rate=62831.73512951655,
    ):
        if self.good:
            if sim is None and self.sim is not None:
                sim = self.sim
            self.ic = sim.copy()
            sim.dt = sim.particles[1].P / np.sqrt(10 * 11)
            sim.exit_max_distance = 100
            if port is not None:
                sim.start_server(port=port)
            airball.tools.rotate_into_plane(sim, "invariable")
            sim.move_to_com()
            sim.integrator = "whckl"
            sim.ri_whfast.safe_mode = False
            sim.ri_whfast.keep_unsynchronized = True
            self.n = sim.N - 1
            self.Nfreq = 2 * self.n if Nfreq is None else Nfreq

            self.N = int(2 ** np.ceil(np.log2(tmax / max_sample_rate)))
            self.t = np.linspace(sim.t, sim.t + tmax, self.N, endpoint=True)

            # self.amd = np.zeros((self.N, 2, self.n))
            self.a = np.zeros((self.N, self.n))
            self.e = np.zeros((self.N, self.n))
            self.inc = np.zeros((self.N, self.n))
            self.z = np.zeros((self.N, self.n), dtype=complex)
            self.ζ = np.zeros((self.N, self.n), dtype=complex)

            for i, t in enumerate(self.t):
                try:
                    sim.integrate(t, exact_finish_time=False)
                except rebound.Escape as _:
                    self.good = False
                    break
                sim.synchronize()
                # self.amd[i] = nAMD(sim)
                orbits = sim.orbits()
                for j in range(self.n):
                    self.a[i, j] = orbits[j].a
                    self.e[i, j] = orbits[j].e
                    self.inc[i, j] = orbits[j].inc
                    self.z[i, j] = orbits[j].e * np.exp(1j * orbits[j].pomega)
                    self.ζ[i, j] = np.sin(orbits[j].inc / 2.0) * np.exp(
                        1j * orbits[j].Omega
                    )
            if port is not None:
                sim.stop_server()
            self.sim = sim
            if self.good:
                # self.AMD = (self.amd[:,0].T/np.sum(self.amd[:,1])).T
                self.gs = []
                self.ss = []
                for i in range(self.n):
                    self.gs.append(fmftc(self.t - self.t[0], self.z[:, i], self.Nfreq))
                    if np.any(self.ζ[:, i]):
                        self.ss.append(
                            fmftc(self.t - self.t[0], self.ζ[:, i], self.Nfreq)
                        )
                    else:
                        self.ss.append(
                            {0: 1j * 0.0}
                        )  # if no inclination, set frequency analysis to 0.

                self.g, self.A, self.Sg = Smatrix(self.gs)
                self.g = (self.g << u.rad / u.yr2pi) << u.arcsec / u.yr

                self.s, self.B, self.Ss = Smatrix(self.ss)
                self.s = (self.s << u.rad / u.yr2pi) << u.arcsec / u.yr

                self.zfft = np.zeros((self.n, self.N), dtype=complex)
                self.ζfft = np.zeros((self.n, self.N), dtype=complex)
                self.freqs = (
                    np.fft.fftfreq(self.N, self.t[1] - self.t[0])
                    * twopi
                    * 360.0
                    * 3600.0
                )
                for k in range(self.n):
                    self.zfft[k] = np.fft.fft(self.z[:, k])
                    self.ζfft[k] = np.fft.fft(self.ζ[:, k])
            else:
                del self.__dict__
                self.sim = sim
                self.good = False

    def modes(self):
        if self.good:
            self.gs = []
            self.ss = []
            for i in range(self.n):
                self.gs.append(fmftc(self.t - self.t[0], self.z[:, i], self.Nfreq))
                if np.any(self.ζ[:, i]):
                    self.ss.append(fmftc(self.t - self.t[0], self.ζ[:, i], self.Nfreq))
                else:
                    self.ss.append(
                        {0: 1j * 0.0}
                    )  # if no inclination, set frequency analysis to 0.

            self.g, self.A, self.Sg = Smatrix(self.gs)
            self.g = (self.g << u.rad / u.yr2pi) << u.arcsec / u.yr

            self.s, self.B, self.Ss = Smatrix(self.ss)
            self.s = (self.s << u.rad / u.yr2pi) << u.arcsec / u.yr
