### all the necessary functions are here
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import sunpy.map
from datetime import datetime
from sunpy.coordinates import frames, sun

# =========================================================
# 1. CREATE SUNPY RADIO MAP
# =========================================================
def make_radio_map(fits_path,
                   observatory_name="NenuFAR (Nançay)",
                   site_lat=47.382*u.deg,
                   site_lon=2.195*u.deg):
    """
    Build a SunPy map from a radio FITS file with proper WCS.
    """

    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data

    # --- flatten data if needed ---
    while data is not None and getattr(data, "ndim", 0) > 2:
        data = np.squeeze(data[0])

    # --- frequency ---
    freq_Hz = hdr.get("CRVAL3", None)
    frequency = (freq_Hz * u.Hz) if freq_Hz is not None else (np.nan * u.Hz)

    # --- pixel scale ---
    cdelt1 = abs(hdr.get("CDELT1", np.nan)) * u.deg
    cdelt2 = abs(hdr.get("CDELT2", np.nan)) * u.deg
    cdelt1 = cdelt1.to(u.arcsec)
    cdelt2 = cdelt2.to(u.arcsec)

    # --- time + observer ---
    obstime = Time(hdr['DATE-OBS'])
    site_loc = EarthLocation(lat=site_lat, lon=site_lon)
    site_gcrs = SkyCoord(site_loc.get_gcrs(obstime))

    # --- reference coordinate ---
    cunit1 = u.Unit(hdr.get("CUNIT1", "deg"))
    cunit2 = u.Unit(hdr.get("CUNIT2", "deg"))

    ref_gcrs = SkyCoord(
        hdr["CRVAL1"] * cunit1,
        hdr["CRVAL2"] * cunit2,
        frame="gcrs",
        obstime=obstime,
        obsgeoloc=site_gcrs.cartesian,
        obsgeovel=site_gcrs.velocity.to_cartesian(),
        distance=site_gcrs.hcrs.distance,
    )

    ref_hpc = ref_gcrs.transform_to(
        frames.Helioprojective(observer=site_gcrs)
    )

    # --- solar rotation correction ---
    P1 = sun.P(obstime)

    ref_pix = np.array([hdr["CRPIX1"] - 1, hdr["CRPIX2"] - 1]) * u.pixel
    scale = np.array([cdelt1.value, cdelt2.value]) * (u.arcsec / u.pixel)

    new_header = sunpy.map.make_fitswcs_header(
        data=data,
        coordinate=ref_hpc,
        reference_pixel=ref_pix,
        scale=scale,
        rotation_angle=-P1,
        wavelength=frequency.to(u.MHz),
        observatory=observatory_name,
    )

    rmap = sunpy.map.Map(data, new_header).rotate()

    return rmap, frequency.to(u.MHz), obstime

# =========================================================
# 4. BATCH PROCESSING (MULTI-FILE)
# =========================================================
def process_multiple_maps(fits_files):
    """
    Process list of FITS files with corresponding shifts.
    Returns list of (map, frequency).
    """

    results = []

    for f in fits_files:
        m, freq, obstime = make_radio_map(f)
        results.append((m, freq, obstime))

    return results

# =========================================================
# 4. Process nenufar spectra .npz data
# =========================================================
def get_data(filename = '/home/sbhunia/check_leakage/data/Q_2024_07_07.npz'):
    cmb_data = np.load(filename)
    data = cmb_data['data']
    #pdb.set_trace()
    data = np.transpose(data)
    data = data[0]
    freq = cmb_data['freq']
    times = [datetime.fromtimestamp(cmb_data['time'][i]) for i in range( cmb_data['time'].shape[0])]
    return data, freq, times
    
# =========================================================
# 4. Plotting NenuFAR contours
# =========================================================
def plot_contours(aia, radio_map, ax, color='white',
                  levels=np.arange(93, 100, 3)*u.percent):

    radio_map.meta['rsun_ref'] = aia.meta['rsun_ref']
    print('start plotting contours')

    with frames.Helioprojective.assume_spherical_screen(aia.observer_coordinate):
        cs = radio_map.draw_contours(axes=ax,levels=levels,colors=color)
    return cs
