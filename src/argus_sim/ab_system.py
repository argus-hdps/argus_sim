"""Conversions between the AB magnitude system and instrument-native units."""

from __future__ import annotations

from typing import Optional, Union

import astropy.constants as c
import astropy.units as u
import numpy as np
import numpy.typing as npt
import scipy.integrate as sing

from . import c as config
from .throughput import SystemThroughput


class ABPhot:
    """A class to handle AB photometry calculations.

    Parameters
    ----------
    throughputs_dir : str
        Directory containing throughput files.
    collecting_area : Optional[u.Quantity[u.cm**2]], optional
        The collecting area of one telescope. If None, the configured ``telescope.collecting_area`` is used.
    throughput_loss : float, optional
        The throughput loss factor. Default is 1.0.

    Attributes
    ----------
    tp : SystemThroughput
        The system throughput object.
    zp_ergs : u.Quantity
        The zero-point flux density in ergs.
    collecting_area : u.Quantity
        The collecting area of the telescope.
    freqs : npt.ArrayLike
        The throughput frequencies.

    Methods
    -------
    photons_from_flux_density(band_name, flux_density)
        Calculate the number of photons from a given flux density.
    photons_from_mag(band_name, mag)
        Calculate the number of photons from a given magnitude.
    photons_from_sed(band_name, wavelength, flux_density)
        Calculate the filter-transmitted photon rate from an arbitrary SED.
    mag_from_sed(band_name, wavelength, flux_density)
        Calculate the synthetic AB magnitude from an arbitrary SED.
    electrons_from_sed(band_name, wavelength, flux_density, airmass, transparency)
        Calculate the detected electron rate from an arbitrary SED through the full system.
    mag_from_photons(band_name, photons)
        Calculate the magnitude from a given number of photons.

    """

    def __init__(
        self,
        throughput: Optional[SystemThroughput] = None,
        throughputs_dir: Optional[str] = None,
        collecting_area: u.Quantity[u.cm**2] | None = None,
        throughput_loss: float = 1.0,
    ) -> None:
        """Initialize the ABPhot object.

        Parameters
        ----------
        throughput : Optional[SystemThroughput], optional
            A pre-constructed SystemThroughput object. If provided,
            throughputs_dir and throughput_loss are ignored.
        throughputs_dir : Optional[str], optional
            Directory containing throughput files. Used only if
            throughput is None.
        collecting_area : Optional[u.Quantity[u.cm**2]], optional
            The collecting area of one telescope. If None, the configured ``telescope.collecting_area`` is used.
        throughput_loss : float, optional
            The throughput loss factor. Default is 1.0.

        """
        if throughput is None:
            self.tp = SystemThroughput(throughputs_dir, throughput_loss=throughput_loss)
        else:
            self.tp = throughput
        # AB zero-point: m_AB = -2.5 log10(F_nu) - 48.60 (Oke & Gunn 1983,
        # ApJ 266, 713, Eq. 7). The constant 48.60 = -2.5 log10(3631 Jy in
        # cgs), so F_nu(m=0) = 3631 Jy = 3.631e-20 erg/s/cm^2/Hz.
        self.zp_ergs = 10 ** (48.60 / -2.5) * (u.erg / (u.second * u.cm**2 * u.Hertz))
        if collecting_area is None:
            collecting_area = config.telescope.collecting_area
        self.collecting_area = collecting_area
        self.freqs = self.tp.freq

    def photons_from_flux_density(
        self,
        band_name: str,
        flux_density: u.Quantity[u.erg / (u.second * u.cm**2 * u.Hertz)],
    ) -> u.Quantity[u.second**-1]:
        """Calculate the number of photons from a given flux density.

        Parameters
        ----------
        band_name : str
            The name of the band.
        flux_density : u.Quantity[u.erg / (u.second * u.cm**2 * u.Hertz)]
            The flux density in ergs per second per square centimeter per Hertz.

        Returns
        -------
        u.Quantity[u.second**-1]
            The number of photons per second.

        """
        # Integrate photon rate: N = A * ∫ (F_nu * R(nu)) / (h * nu) dnu
        # where R(nu) is the band response and h*nu is the photon energy
        resp_func = self.tp.as_callable(
            self.freqs,
            self.tp.filt_freq[band_name],
        )
        y = (flux_density * resp_func(self.freqs) / (c.h * self.freqs)).cgs.value
        x = self.freqs.to(u.Hertz).value
        photon_density = (
            sing.trapezoid(
                y,
                x,
                dx=np.mean(np.diff(x)),
            )
            * u.Hertz
            * u.cm ** (-2)
        )
        return (photon_density).cgs * self.collecting_area * u.photon

    def _sed_flux_lambda_on_grid(
        self,
        band_name: str,
        wavelength: u.Quantity,
        flux_density: u.Quantity,
        allow_partial: bool = False,
    ) -> np.ndarray:
        """Interpolate an input SED onto the system wavelength grid.

        The input flux density may be expressed per unit wavelength (for
        example, ``erg / s / cm2 / Angstrom``) or per unit frequency (for
        example, ``Jy``). It is converted to ``F_lambda`` before interpolation.
        Uncovered filter support raises unless ``allow_partial=True``.

        Parameters
        ----------
        band_name : str
            Name of the loaded filter band.
        wavelength : astropy.units.Quantity
            One-dimensional wavelength array.
        flux_density : astropy.units.Quantity
            One-dimensional spectral flux-density array sampled at
            ``wavelength``.
        allow_partial : bool, optional
            If False (default), require the input SED to cover the full
            non-zero support of the requested filter. If True, uncovered
            wavelengths contribute zero flux.

        Returns
        -------
        numpy.ndarray
            ``F_lambda`` sampled on ``SystemThroughput.wav`` in
            erg s-1 cm-2 nm-1.

        """
        if band_name not in self.tp.filt_wav:
            raise ValueError(f"Filter {band_name!r} is not loaded in SystemThroughput.")

        if not isinstance(wavelength, u.Quantity) or not isinstance(flux_density, u.Quantity):
            raise TypeError("wavelength and flux_density must be astropy Quantity arrays.")
        if wavelength.ndim != 1 or flux_density.ndim != 1:
            raise ValueError("wavelength and flux_density must be one-dimensional arrays.")
        if wavelength.shape != flux_density.shape:
            raise ValueError("wavelength and flux_density must have the same shape.")
        if wavelength.size < 2:
            raise ValueError("An SED must contain at least two wavelength samples.")
        if not wavelength.unit.is_equivalent(u.nm):
            raise u.UnitConversionError("wavelength must carry physical wavelength units.")

        order = np.argsort(wavelength.to_value(u.nm))
        wavelength = wavelength[order]
        flux_density = flux_density[order]
        wav_nm = wavelength.to_value(u.nm)
        if not np.all(np.isfinite(wav_nm)) or np.any(wav_nm <= 0):
            raise ValueError("wavelength must contain finite, positive values.")
        if np.any(np.diff(wav_nm) <= 0):
            raise ValueError("wavelength samples must be unique.")

        target_unit = u.erg / (u.second * u.cm**2 * u.nm)
        try:
            flux_lambda = flux_density.to(target_unit, equivalencies=u.spectral_density(wavelength))
        except u.UnitConversionError as exc:
            raise u.UnitConversionError(
                "flux_density must be a spectral flux density (for example F_lambda, F_nu, or Jy)."
            ) from exc

        flux_values = flux_lambda.to_value(target_unit)
        if not np.all(np.isfinite(flux_values)):
            raise ValueError("flux_density must contain only finite values.")

        filter_response = np.asarray(self.tp.filt_wav[band_name], dtype=float)
        support = filter_response > 0
        if not np.any(support):
            raise ValueError(f"Filter {band_name!r} has no non-zero throughput.")
        system_wav_nm = self.tp.wav.to_value(u.nm)
        # Unit conversion can move an endpoint by a few floating-point ulps.
        # Snap only that numerical roundoff so an identical grid in Angstroms
        # neither fails coverage nor loses its endpoint during interpolation.
        for endpoint in (0, -1):
            nearest = system_wav_nm[np.argmin(np.abs(system_wav_nm - wav_nm[endpoint]))]
            if abs(wav_nm[endpoint] - nearest) <= 8 * np.finfo(float).eps * abs(nearest):
                wav_nm[endpoint] = nearest
        if not allow_partial:
            support_min = system_wav_nm[support][0]
            support_max = system_wav_nm[support][-1]
            if wav_nm[0] > support_min or wav_nm[-1] < support_max:
                raise ValueError(
                    f"SED wavelength coverage ({wav_nm[0]:.1f}-{wav_nm[-1]:.1f} nm) does not cover the full "
                    f"{band_name!r} band ({support_min:.1f}-{support_max:.1f} nm). "
                    "Pass allow_partial=True to integrate the overlapping region only."
                )

        return np.interp(system_wav_nm, wav_nm, flux_values, left=0.0, right=0.0)

    def photons_from_sed(
        self,
        band_name: str,
        wavelength: u.Quantity,
        flux_density: u.Quantity,
        *,
        allow_partial: bool = False,
    ) -> u.Quantity[u.photon / u.second]:
        """Calculate the filter-transmitted photon rate from an arbitrary SED.

        This is the SED analogue of :meth:`photons_from_flux_density`. The
        result includes the telescope collecting area and filter transmission,
        but not optics, detector QE, or atmospheric extinction. This makes the
        returned rate compatible with :meth:`mag_from_photons`.

        Parameters
        ----------
        band_name : str
            Name of the loaded filter band.
        wavelength : astropy.units.Quantity
            Wavelength samples for the SED.
        flux_density : astropy.units.Quantity
            Observer-frame spectral flux density above the atmosphere. Both
            F_lambda and F_nu-like units are accepted.
        allow_partial : bool, optional
            Allow the SED to cover only part of the filter band. Uncovered
            wavelengths are treated as zero flux. Default is False.

        Returns
        -------
        astropy.units.Quantity
            Filter-transmitted photon rate in photon s-1.

        """
        flux_lambda = self._sed_flux_lambda_on_grid(band_name, wavelength, flux_density, allow_partial)
        photon_energy_erg = (c.h * c.c / self.tp.wav).to_value(u.erg)
        response = np.asarray(self.tp.filt_wav[band_name], dtype=float)
        integrand = flux_lambda * response / photon_energy_erg
        rate = sing.trapezoid(integrand, self.tp.wav.to_value(u.nm)) * self.collecting_area.to_value(u.cm**2)
        return rate * u.photon / u.second

    def mag_from_sed(
        self,
        band_name: str,
        wavelength: u.Quantity,
        flux_density: u.Quantity,
        *,
        allow_partial: bool = False,
    ) -> float:
        """Calculate a synthetic AB magnitude by integrating an arbitrary SED.

        ``wavelength`` and ``flux_density`` are Quantity arrays giving the
        observer-frame source flux above the atmosphere. Both F_lambda and
        F_nu-like units are accepted. By default the SED must cover the full
        non-zero filter support; ``allow_partial=True`` treats uncovered
        wavelengths as zero flux. The filter-only photon rate is calibrated
        with the same AB zero point as :meth:`mag_from_photons`.
        """
        photons = self.photons_from_sed(band_name, wavelength, flux_density, allow_partial=allow_partial)
        return self.mag_from_photons(band_name, photons)

    def electrons_from_sed(
        self,
        band_name: str,
        wavelength: u.Quantity,
        flux_density: u.Quantity,
        *,
        airmass: float = 1.0,
        transparency: float = 1.0,
        allow_partial: bool = False,
    ) -> u.Quantity[u.electron / u.second]:
        """Calculate the detected electron rate from an arbitrary SED.

        The SED is integrated through the requested filter and the complete
        Argus response: optics throughput, detector QE, and atmospheric
        transmission raised to ``airmass``. ``transparency`` is an additional
        grey attenuation factor, matching the survey simulator's cloud model.

        Unlike the pre-tabulated AB-source signal throughput used by the survey
        depth calculation, this method keeps the source SED inside the
        wavelength integral. It therefore preserves source-colour dependence.

        Parameters
        ----------
        band_name : str
            Name of the loaded filter band.
        wavelength : astropy.units.Quantity
            Wavelength samples for the SED.
        flux_density : astropy.units.Quantity
            Observer-frame spectral flux density above the atmosphere. Both
            F_lambda and F_nu-like units are accepted.
        airmass : float, optional
            Atmospheric airmass. Default is 1.0. Use 0.0 for no atmospheric
            extinction.
        transparency : float, optional
            Additional grey atmospheric transparency factor in [0, 1].
            Default is 1.0.
        allow_partial : bool, optional
            Allow the SED to cover only part of the filter band. Uncovered
            wavelengths are treated as zero flux. Default is False.

        Returns
        -------
        astropy.units.Quantity
            Detected source electron rate in electron s-1, suitable for
            :meth:`argus_sim.NoiseBudget.get_snr`.

        """
        if not np.isscalar(airmass) or not np.isfinite(airmass) or airmass < 0:
            raise ValueError("airmass must be a finite scalar >= 0.")
        if not np.isscalar(transparency) or not np.isfinite(transparency) or not 0 <= transparency <= 1:
            raise ValueError("transparency must be a finite scalar in [0, 1].")

        flux_lambda = self._sed_flux_lambda_on_grid(band_name, wavelength, flux_density, allow_partial)
        photon_energy_erg = (c.h * c.c / self.tp.wav).to_value(u.erg)
        response = (
            np.asarray(self.tp.filt_wav[band_name], dtype=float)
            * float(self.tp.optics[band_name])
            * np.asarray(self.tp.qe_wav, dtype=float)
            * np.asarray(self.tp.filt_wav.atmosphere, dtype=float) ** float(airmass)
            * float(transparency)
        )
        integrand = flux_lambda * response / photon_energy_erg
        rate = sing.trapezoid(integrand, self.tp.wav.to_value(u.nm)) * self.collecting_area.to_value(u.cm**2)
        return rate * u.electron / u.second

    def photons_from_mag(self, band_name: str, mag: Union[float, npt.ArrayLike]) -> u.Quantity[u.second**-1]:
        """Calculate the number of photons from a given magnitude.

        Parameters
        ----------
        band_name : str
            The name of the band.
        mag : float
            The magnitude.

        Returns
        -------
        u.Quantity[u.second**-1]
            The number of photons per second.

        """
        zp_photons = self.photons_from_flux_density(band_name, self.zp_ergs)
        return zp_photons * 10 ** (mag / -2.5)

    def mag_from_photons(self, band_name: str, photons: u.Quantity[u.second**-1]) -> float:
        """Calculate the magnitude from a given number of photons.

        Parameters
        ----------
        band_name : str
            The name of the band.
        photons : u.Quantity[u.second**-1]
            The number of photons per second.

        Returns
        -------
        float
            The magnitude.

        """
        zp_photons = self.photons_from_flux_density(band_name, self.zp_ergs)
        return -2.5 * np.log10(photons / zp_photons)
