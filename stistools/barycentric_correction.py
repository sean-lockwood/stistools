#!/usr/bin/env python

import time
import shutil
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy import units as u, constants as c
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body_barycentric
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.coordinates import ITRS, GCRS, ICRS
from astroquery.jplhorizons import Horizons


__doc__ = """
This task :func:`barycentric_correction' calculates the barycentric correction
between HST and the Solar System barycenter for times in HST/STIS observations.
It updates the times in uncalibrated (tag, raw) and calibrated data. It stores
the "old" times in a new column for tag data and in new header keywords for
raw and calibrated data. The function puts the various delay terms and
positions in the header as well.

The function works by first determining the position of the Earth with
Astropy and HST with either JPL Horizons or the STScI-provided HST
orbital files.

For *tag.fits files with many times to convert, performance can be slow due to
the high number of coordinate transformations.

Stand-alone tasks :func:`calc_delay_jpl' :func:`calc_delay_orbfile' can be used
to calculate barycentric corrections without file modifications.

Task :func:`odelay_file_compare' shows differences in times between two STIS
.fits files. This is useful in what barycentric corrections were
calculated.

Examples
--------

:func:`barycentric_correction` with default values:

>>> import stistools
>>> stistools.barycentric_correction.barycentric_correction("oep502040_x1d.fits")
"""

__taskname__ = "barycentric_correction"
__version__ = "1.0"
__vdate__ = "21-August-2025"
__author__ = "J. Lothringer"

# TODO clean up the global variables if desired, only SECPERDAY is used in this file
# File types
EVENTS_TABLE = 1

SECPERDAY = 86400  #D0  # number of sec in a day

# Replacing with Astropy unit values
# Keeping old ones commented out for comparison
#LYPERPC = 3.261633   #D0  # light years per parsec
LYPERPC = u.pc.to('lightyear')
#KMPERPC  = 3.085678e13  #D13  # kilometers per parsec
KMPERPC = u.pc.to('km')
#AUPERPC  = 206264.8062470964  #D0  # how many AU in one parsec
AUPERPC = u.pc.to('AU')
KMPERAU = KMPERPC / AUPERPC
#CLIGHT = 499.00479   #D0  # light travel time (in sec) of 1 AU
CLIGHT = (u.AU / c.c).to('s').value

#EARTH_EPHEMERIS = 1
#OBS_EPHEMERIS = 2

JD_TO_MJD = 2400000.5   #d0  # subtract from JD to get MJD

class OrbFileError(ValueError):
    """Orbit file does not cover the requested time range.
    """
    pass


def barycentric_correction(table_names, verbose=True, distance=1e9,
                           hst_orb=None, in_col='TIME',
                           time_script=False, outfiles=None):
    """ Calculate barycentric corrections for HST's position.

        Calculates time-delay barycentric corrections from HST's position
        to the Solar System barycenter. This correction includes the classic
        geometric Roemer delay, as well as the general relativistic Einstein
        delay.

        This function uses modern Astropy tools to calculate the position
        of the barycenter, Earth's location, and deal with the time standards.
        The function has been tested against a Python implementation of
        the previous odelaytime IRAF task and found to be consistent.
        We have also tested against other barycentric correction tools,
        including barycorr, astroutils, and pintbary.

        HST's changing location around the Earth can lead to time-delay
        differences of up to ~46 milliseconds. HST's location can be
        determined either through STScI-provided HST orbital files, or through
        a query to JPL Horizons.

        These calculations are accurate to within 1 millisecond
        outside the Solar System, and to within 5 milliseconds inside
        the Solar System.

        Parameters
        ----------
        table_names: list[str]
            List of strings with the file names to be time-corrected.

        verbose: bool
            Prints completion messages during execution.

        distance: float
            Distance the object is from HST in AU. Default is a trillion
            AU. Most important for objects in our Solar System as it
            is repsonsible for second-order correction, up to minutes.
            At 1 parsec, the correction can be on the order of a few ms.

        hst_orb: str
            Name of HST orbital file (generally starts p, ends as .fit) that
            covers the time of the observations. If not provided, JPL Horizons
            will be used to get HST's orbital position.

        in_col: str
            Usually 'TIME' or 'time'. Used for compatability with files where
            the column name for time is upper- or lower-case.

        time_script: bool
            Set to True if you want to time how long the script takes. Useful
            for debugging, especially for .tag files with large numbers of
            events.

        outfiles: list
            Default None. If not None, then it is a list of output files for
            each table_names. Each table_name will be copied over to the corr-
            esponding outfile name.

        Returns
        -------
        None
            Nothing is returned directly, but the file is written to output.
    """

    if time_script:
        tstart = time.time()

    # TODO need to address errors if single input file and outfile name are submitted
    for ii, in_table_file in enumerate(table_names):

        if verbose:
            print(f"odelaytime: processing {in_table_file} ...")

        # Copy to new outfile, otherwise overwrite input file.
        # TODO what should happen if the length of table_names and outfiles don't match?
        filename = in_table_file
        if outfiles is not None:
            filename = outfiles[ii]
            if verbose:
                print(f"Copying {in_table_file} to {filename}")
            shutil.copy(in_table_file, filename)

        in_hdul = fits.open(filename, mode='update')

        # determine the file type, based on the first extension
        # original code (ln 124) contains some kind of catch all
        # for SCI_IMAGE? It also caught anything outside of those options,
        # might want to add that back in as well.
        extname = in_hdul[1].header['EXTNAME']
        if extname not in ["EVENTS", "SCI"]:
            raise ValueError(f'Unexpected extension name:  {extname}')

        if 'DELAYCOR' in in_hdul[0].header and \
                        in_hdul[0].header['DELAYCOR'] == "COMPLETE":
            print(f"{in_table_file} has already been corrected, so no further processing"
                  " will be applied to this file")
            continue

        in_hdul[0].header['DELAYCOR'] = "PERFORM"

        # COS has no TEXPSTRT in primary header, so use EXPSTART in first ext
        # TODO removed mjd2 since it isn't used, check that this is okay
        if in_hdul[0].header['INSTRUME'] == "STIS":
            mjd1 = in_hdul[0].header['TEXPSTRT']
        elif in_hdul[0].header['INSTRUME'] == "COS":
            mjd1 = in_hdul[1].header['EXPSTART']
        else:
            raise ValueError('baycentric_correction only works with STIS and COS files.')

        ra = in_hdul[0].header['RA_TARG']
        dec = in_hdul[0].header['DEC_TARG']

        if 'NEXTEND' in in_hdul[0].header:
            nextend = in_hdul[0].header['NEXTEND']
        else:
            nextend = 1

        if extname == "EVENTS":
            # assume (for now) that the last extension is a GTI table
            nevents_tab = max(nextend - 1, 1)
            nsci_ext = 0
        elif extname == "SCI":
            nevents_tab = 0
            nsci_ext = nextend
        else:
            nevents_tab = 0
            nsci_ext = nextend


        # Get the delay at the exposure start time.  We'll subtract this
        # delay from each time that we update, if that time is relative
        # to EXPSTART (which must be the same as TEXPSTRT).
        #t0_delay = all_delay(mjd1, parallax, objvec, earth_ephem_table,
        #                     len(earth_ephem_table), obs_ephem_table, npts_obs)

        t0_delay = calc_delay(mjd1, ra, dec, hst_orb, distance=distance)

        if extname == "EVENTS":
            # update times in GTI table
            if 'GTI' in in_hdul:
                gti_tab = in_hdul['GTI'].data
                for row in gti_tab:
                    for key in ['START', 'STOP']:
                        tm = row[key]
                        epoch = mjd1 + tm / SECPERDAY
                        delta_sec = calc_delay(epoch, ra, dec, hst_orb, distance=distance)
                        row[key] = tm + (delta_sec - t0_delay).value * SECPERDAY

                in_hdul.flush()

                if verbose:
                    print("GTI extension has been updated")

            else:
                # If there's no GTI, last extensions is an EVENTS table
                nevents_tab += 1

        if time_script:
            tcheck1 = time.time()
            print(f'Checkpoint 1: {tcheck1 - tstart} s')

        # loop through all EVENTS extensions (there will be none if
        # the input is an x1d or image file)
        for e_indx in range(1, nevents_tab + 1):
            events_tab = in_hdul['EVENTS', e_indx]
            mjd1 = events_tab.header['EXPSTART']

            # find the time (since EXPSTART) column in the table
            nrows = len(events_tab.data[in_col])

            # pull out just the time array
            time_array = events_tab.data[in_col]

            # go through each row
            if verbose:
                print(f"Number of times in extension: {nrows}")

            #re-written to do all rows at once, so we can interpolate times,
            #instead of calling Horizons 4.5 million times
            epoch_array = mjd1 + time_array / SECPERDAY
            # TODO do we need to pass in_col here?
            delta_sec = calc_delay(epoch_array, ra, dec, hst_orb, distance=distance)

            # This is correcting for the difference between the new time interval
            # and the *corrected* exposure start time
            # (i.e., the add'l correction now that the Earth and HST have moved)
            time_array = time_array + (delta_sec - t0_delay).value * SECPERDAY

            # replace time array <- wasn't being restuffed before
            in_hdul['EVENTS', e_indx].data[in_col] = time_array

            # add delaytime to EXPSTART and EXPEND, and update header
            for key in ['EXPSTART', 'EXPEND']:
                mjd = events_tab.header[key]
                # TODO do we need to pass in_col here?
                delta_sec = calc_delay(mjd, ra, dec, hst_orb, distance=distance)
                events_tab.header[key] = mjd + delta_sec.value

            in_hdul.flush()
            if verbose:
                print(f"    [EVENTS,{e_indx}] extension has been updated")

        # Loop through all x1d or image extensions (there will be none if
        # the input is an events file).  Note that we only expect EXPSTART
        # and EXPEND to be present in SCI extensions; however, they
        # could be in ERR and DQ as well (e.g. in output from inttag),
        # and if they are present they must be updated.
        for e_indx in range(1, nsci_ext + 1):
            cur_tab = in_hdul[e_indx]

            modified = False

            # add delaytime to EXPSTART and EXPEND, and update header
            for key in ['EXPSTART', 'EXPEND']:
                if key in cur_tab.header:
                    mjd = cur_tab.header[key]
                    delta_sec = calc_delay(mjd, ra, dec, hst_orb, distance=distance)
                    cur_tab.header[key] = mjd + delta_sec.value
                    modified = True

            in_hdul.flush()
            if verbose and modified:
                print(f"    extension {e_indx} has been updated")

        if time_script:
            tcheck2 = time.time()

            print(f'Checkpoint 2: {tcheck2 - tstart} s')

        # IF STIS
        # add delaytime to TEXPSTRT and TEXPEND, and update primary header
        # COS has no TEXPSTRT in primary header
        if in_hdul[0].header['INSTRUME'] == "STIS":
            for key in ['TEXPSTRT', 'TEXPEND']:
                mjd = in_hdul[0].header[key]
                delta_sec = calc_delay(mjd, ra, dec, hst_orb, distance=distance)

                in_hdul[0].header[key] = mjd + delta_sec.value

        # add keyword to flag the fact that the times have been corrected
        in_hdul[0].header['DELAYCOR'] = ("COMPLETE",
                                         "delaytime has been applied")
        in_hdul[0].header['history'] = "Times corrected to solar system barycenter;"
        in_hdul[0].header['history'] = "All times now in BJD_TDB;"

        if hst_orb is not None:
            in_hdul[0].header['history'] = f"HST orbital table {hst_orb}"
        else:
            in_hdul[0].header['history'] = "no HST orbital tables were used."


        # close in file
        in_hdul.close()

        if verbose:
            print("... done")

        if time_script:
            tcheck3 = time.time()
            print(f'Checkpoint 3: {tcheck3 - tstart} s')


def calc_delay(times, ra, dec, hst_orb=None, distance=1e9, in_col='Time', verbose=True):
    """Calculate the light-travel time correction for HST observations.

    This function computes the barycentric light-travel time delay for the Hubble Space Telescope (HST)
    given a set of observation times and a target sky position (RA, Dec).

    If an HST orbit file is provided, it interpolates HST’s position from a provided orbit file, combines
    it with Earth's barycentric position, and calculates the light-travel time correction to the Solar
    System barycenter, including a finite-distance correction term.

    If an HST orbit file is not provided, this function queries JPL Horizons for HST's geocentric
    state-vector (or interpolating a regular-sampled vector set) and combining it with Earth's
    barycentric position. A finite-distance correction is applied to account for targets that are
    not at infinite distance.

    Parameters
    ----------
    times : array-like or float
        Observation times in Modified Julian Date (MJD), corresponding to HST exposures.

    ra : float
        Right ascension of the target in degrees.

    dec : float
        Declination of the target in degrees.

    hst_orb : str, optional
        Path to the HST orbit FITS file. This file must contain columns `TIME`, `X`, `Y`, and `Z`
        giving HST’s position (in km) relative to the Earth's center.

    distance : float, optional
        Distance to the target in kilometers (default is `1e9`, effectively infinite distance).
        Used to apply the finite-distance light-travel time correction.

    in_col : str, optional
        If orbital file uses something other than 'Time' for the time axis, replace
        with the correct column name.

    verbose : bool, optional
        If True (default), print information about the finite-distance correction
        and the calculated light-travel times.

    Returns
    -------
    lt_time : `~astropy.units.Quantity`
        Light-travel time correction(s) (Astropy Quantity) in days. This is the time
        that should be added to the observation times to obtain barycentric arrival times,
        and includes the finite-distance correction.

    Notes
    -----
    - Uses the JPL planetary ephemeris (``jpl``) via Astropy's ``solar_system_ephemeris``
      for consistency with Horizons.
    - The finite-distance correction implemented here follows the same algebraic
      form used elsewhere in this codebase:
        correction_term = (-0.5 / D) * (|r|^2 - (n·r)^2) / c
      where `r` is the HST barycentric position vector, `n` is the unit vector toward
      the target, `D` is the provided `distance`, and `c` is the speed of light.
      The expression is converted into days before being returned.
    - Requires `astroquery` (Horizons), Astropy with the `jplephem` ephemeris available,
      and a working internet connection for Horizons queries when not using a local orbit file.

    Examples
    --------
    >>> # single time (MJD), RA/Dec in degrees
    >>> barycentric_correction.calc_delay_jpl(55521.123, 210.8023, -47.393)
    Finite distance correction: -7.804152203899432e-08 s
    Light travel times: -405.56807173759836 s

    >>> # multiple times
    >>> times = [60200.123, 60200.124, 60200.125]
    >>> lt_array = calc_delay_jpl(times, 210.8023, -47.393, distance=1e9, verbose=False)

    >>> # Using an HST orbit file
    >>> barycentric_correction.calc_delay([55521.123], 210.8023, -47.393, hst_orb='pubj0000r.fit')
    Finite distance correction: [-7.80563708e-08] s
    Light travel times: [-405.56687377] s
    """

    has_hst_orb = False
    if hst_orb is not None:
        has_hst_orb = True

    # Using the JPL epehermis to be consistent with what Horizons gives
    # see https://github.com/astropy/astropy/pull/11608
    # Will require jplephem package, but that's already in stenv!
    # TODO need to add jplephem to package requirements
    solar_system_ephemeris.set('jpl')

    # Put times into Astropy time object
    # We will query HST's position at this time
    # We will be off by the HST-geocenter light travel time, FWIW
    times_geo = Time(times,
                     format='mjd',
                     scale='utc',
                     location=EarthLocation.\
                     from_geocentric(0, 0, 0, unit='m'))


    # Get HST's location
    hstarr = get_hst_location(times, times_geo, hst_orb=hst_orb, in_col=in_col, verbose=verbose)

    # Caclculate HST's position relative to the solar system barycenter
    # and determine it's cartesian ITRS coordinates
    hstbary, itrs_coordinates = calculate_hst_coordinates(times_geo, hstarr, has_hst_orb=has_hst_orb)

    # Define our targets location on the sky
    target = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')

    # Put into an array to dot with hstbary
    # Cartesian makes this a unit vector in direction of target
    target_arr = [target.cartesian.x.value,
                target.cartesian.y.value,
                target.cartesian.z.value]

    # Calculate the finite-distance correction term
    if has_hst_orb:
        correction_term = ((-0.5 / distance) *
                        (np.sum((np.array(hstbary))**2) -
                            (np.dot(target_arr, hstbary))**2) * u.AU /
                        c.c).to('day')
    else:
        # Actually just need to make the above sum over the correction axis
        # if hstbary has multiple points
        # TODO check if the multiple points handling (setting sum axis=0)
        # needs to be done for the HST orb file case too
        correction_term = ((-0.5 / distance) *
                          (np.sum((np.array(hstbary))**2, axis=0) -
                            (np.dot(target_arr, hstbary))**2) * u.AU /
                          c.c).to('day')

    if verbose:
        if correction_term.size < 10:
            print(f"Finite distance correction: {correction_term.to('s')}")
        else:
            print(f"Finite distance correction: \
                    {correction_term[0:10].to('s')}")

    # Let's now define HST's location relative to the barycenter
    # from_geocentric() is expecting an ITRS coordinate!

    with erfa_astrom.set(ErfaAstromInterpolator(1000 * u.s)):
        hstloc = EarthLocation.from_geocentric(x=itrs_coordinates.x,
                                                y=itrs_coordinates.y,
                                                z=itrs_coordinates.z)
        # Define the times, now with the correct location
        hsttime = Time(times, format='mjd', scale='utc',
                        location=hstloc)

        # Calculate the light travel time,
        # adding the correction term above.
        # Then define the new barycenter times!
        lt_time = hsttime.light_travel_time(target) + correction_term

    if verbose:
        if lt_time.size < 10:
            print(f"Light travel times: {lt_time.to('s')}")
        else:
            print(f"Light travel times: {lt_time[0:10].to('s')}")
    #ssbtimes = hsttime.tdb + lt_time

    return lt_time

def get_hst_location(times, times_geo, hst_orb=None, in_col='Time', verbose=True):
    """Get HST's location from an HST orbit file or a JPL Horizon query.

    Parameters
    ----------
    times : array-like or `~astropy.time.Time`
        Observation times in Modified Julian Date (MJD). Can be a scalar or an array.

    times_geo : array-like or `~astropy.time.Time`
        Geocentric observation times.

    hst_orb : str, optional
        Path to the HST orbit FITS file. This file must contain columns `TIME`, `X`, `Y`, and `Z`
        giving HST’s position (in km) relative to the Earth's center.

    in_col : str, optional
        If orbital file uses something other than 'Time' for the time axis, replace
        with the correct column name.

    verbose : bool, optional
        If True (default), print additional status information.

    Returns
    -------
    hstarr : `array-like, ~astropy.units.Quantity`
        HST position vector stored as a list of cartesian coordinates
        in astropy units of AU.

    Raises
    ------
    OrbFileError
        If the orbit file does not cover the requested time range.

    Exception
        If the Horizons query or interpolation fails to produce vectors covering the
        requested time range, or if the vector transformation cannot be performed.

    Notes
    -----
    - When using an HST orbit file, interpolates HST’s orbital position at each observation time using cubic interpolation
      to avoid excessive Horizons queries.  The resulting correction term accounts for HST’s motion relative to the Solar System barycenter.
    - For multiple times the function queries Horizons for HST (-48, location 500)
      with a regular step (1 minute) over the time range (plus a 5-minute margin)
      and cubic-interpolates the returned vectors to the requested times. For a single
      scalar time it queries Horizons directly at that epoch.
    - Requires `astroquery` (Horizons), Astropy with the `jplephem` ephemeris available,
      and a working internet connection for Horizons queries when not using a local orbit file.
    """

    # If using an HST ORB file
    if hst_orb is not None:
        with fits.open(hst_orb) as hdu_orb:
            f_hstx = interp1d(float(hdu_orb[1].header['FIRSTMJD']) + hdu_orb[1].data[in_col].astype(np.float64) / SECPERDAY, hdu_orb[1].data['X'], kind='cubic')
            f_hsty = interp1d(float(hdu_orb[1].header['FIRSTMJD']) + hdu_orb[1].data[in_col].astype(np.float64) / SECPERDAY, hdu_orb[1].data['Y'], kind='cubic')
            f_hstz = interp1d(float(hdu_orb[1].header['FIRSTMJD']) + hdu_orb[1].data[in_col].astype(np.float64) / SECPERDAY, hdu_orb[1].data['Z'], kind='cubic')

        try:
            hstvecx = f_hstx(times_geo.tdb.mjd)
            hstvecy = f_hsty(times_geo.tdb.mjd)
            hstvecz = f_hstz(times_geo.tdb.mjd)
        except ValueError as e:
            raise OrbFileError('Orbit file does not match observation times.  '
                            'Make sure you have got the correct one.') from e

        hstarr = [hstvecx, hstvecy, hstvecz] * u.km

    # If no HST ORB file
    # Interpolate HST's position if more than just one time:
    # We can't query all times to Horizons because there's
    # too many. Let's get HST's position every minutes instead,
    # and then interpolate...
    # Position every minute should be good enough. If we were
    # off by a minute, that would be a max of 6ms light travel time
    # So with interpolation, we'll be dandy.
    # Adding one minute to each side to avoid extrapolating
    elif np.isscalar(times) is False:
        if verbose:
            print('Multiple times: interpolating positions')
        # Query five minute before and after min and max times
        # (In case we only query a minute of tags, we need more for spline)
        # With a step of 1 min
        epochs = {'start':(np.min(times_geo.tdb) - (5. * u.min)).\
                  to_datetime().strftime("%Y-%m-%d %H:%M:%S"),
                  'stop':(np.max(times_geo.tdb) + (5. * u.min)).\
                  to_datetime().strftime("%Y-%m-%d %H:%M:%S"),
                  'step':'1m'}
        hstobj = Horizons(id='-48', location='500',
                          epochs=epochs)

        # Must be refplane = 'earth' not 'ecliptic
        hstvec = hstobj.vectors(refplane='earth')

        # Run interpolation
        f_hstx = interp1d(hstvec['datetime_jd'], hstvec['x'], kind='cubic')
        f_hsty = interp1d(hstvec['datetime_jd'], hstvec['y'], kind='cubic')
        f_hstz = interp1d(hstvec['datetime_jd'], hstvec['z'], kind='cubic')

        hstvecx = f_hstx(times_geo.tdb.jd)
        hstvecy = f_hsty(times_geo.tdb.jd)
        hstvecz = f_hstz(times_geo.tdb.jd)

        # Re-package witn units
        hstarr = [hstvecx, hstvecy, hstvecz] * u.AU

    else:
        # Query HST's gencentric position in JPL Horizons
        hstobj = Horizons(id='-48', location='500',
                          epochs=times_geo.tdb.jd)

        # Must be refplane = 'earth' not 'ecliptic
        hstvec = hstobj.vectors(refplane='earth')

        # Re-package witn units
        hstarr = [hstvec['x'][0], hstvec['y'][0], hstvec['z'][0]] * u.AU

    return hstarr

def calculate_hst_coordinates(times_geo, hstarr, has_hst_orb=False):
    """Calculate HST's coordinates relative to the Solar System barycenter.

    Parameters
    ----------
    times : array-like or `~astropy.time.Time`
        Geocentric observation times in Modified Julian Date (MJD).
        Can be a scalar or an array.

    hstarr : `array-like, ~astropy.units.Quantity`
        HST position vector stored as a list of cartesian coordinates
        in astropy units of AU.

    has_hst_orb : bool, optional
        Flag to indicate whether the HST position vector in hstarr was
        taken from an HST orbit file, by default False

    Returns
    -------
    hstbary : list
        Vector of HST's position relative to the Solar System barycenter.

    hst_itrs_coordinates : `~astropy.coordinates.builtin_frames.itrs.ITRS`
        ITRS frame coordinates of HST's location relative to the Solar
        System barycenter.

    Notes
    -----
    - The HST vector returned from Horizons is treated as geocentric and then transformed
      to ITRS/ICRS and added to Earth's barycentric position from Astropy's
      ``get_body_barycentric('earth', ...)`` to obtain HST's barycentric position.
    """

    with erfa_astrom.set(ErfaAstromInterpolator(1000 * u.s)):

        # Get Earth's position
        # Can also take a few minutes with lots of times
        earthICRS = get_body_barycentric('earth', times_geo)

        if has_hst_orb:
            # Make into ICRS object at proper times
            hstICRS = ICRS([hstarr[0],
                            hstarr[1],
                            hstarr[2]],
                            representation_type='cartesian')

            # Add the two vectors together to get HST's position relative
            # to the Solar System barycenter
            # We keep this in an array format to calculate the correction term,
            # but convert to ICRS coordinate for light travel times.
            hstbary = [earthICRS.x.to('AU').value + hstICRS.x.to('AU').value,
                    earthICRS.y.to('AU').value + hstICRS.y.to('AU').value,
                    earthICRS.z.to('AU').value + hstICRS.z.to('AU').value]

            hstbary_icrs = ICRS(hstbary * u.AU, representation_type='cartesian')

            # from_geocentric() is expecting an ITRS coordinate, so we need to convert.
            hstbary_itrs = hstbary_icrs.transform_to(ITRS(obstime=times_geo))
            hst_itrs_coordinates = hstbary_itrs.cartesian
        else:
            # Make into GCRS object at proper times
            hstGCRS = GCRS([hstarr[0],
                            hstarr[1],
                            hstarr[2]],
                        representation_type='cartesian',
                        obstime=times_geo)

            # Convert to ICRS so we can add to Earth's barycentric location
            # This takes a bit of time if there are many times to convert
            hst_itrs_coordinates = hstGCRS.transform_to(ITRS(obstime=times_geo))

            # Add the two vectors together to get HST's position relative
            # to the Solar System barycenter
            hstbary = [earthICRS.x.to('AU').value + hst_itrs_coordinates.x.value,
                    earthICRS.y.to('AU').value + hst_itrs_coordinates.y.value,
                    earthICRS.z.to('AU').value + hst_itrs_coordinates.z.value]

    return hstbary, hst_itrs_coordinates


def odelay_file_compare(file1, file2):
    """Compare timing information between two FITS files.

    Computes and prints the differences in exposure start times and data timestamps
    between two FITS files, typically used for verifying time coordinate consistency
    in astronomical observations.

    Parameters
    ----------
    file1 : str
        Path to the first FITS file.

    file2 : str
        Path to the second FITS file to compare against file1.

    Returns
    -------
    None
        This function only prints comparison results to stdout.

    Notes
    -----
    The function prints the following comparisons (file2 - file1):
    - TEXPSTRT header keyword difference in days and seconds
    - EXPSTART header keyword difference in days and seconds (from extension 1)
    - First and last TIME values difference in seconds (only if 'tag' is in file1 name)
    - TT to TDB time scale conversion difference for reference

    The function assumes:
    - Both files have TEXPSTRT in the primary header (extension 0)
    - Both files have EXPSTART in extension 1 header
    - If 'tag' appears in file1 path, extension 1 contains a table with the
      time column specified by in_col. The script will print the difference
      between the first and last times in each file.

    Examples
    --------
    >>> odelay_file_compare('observation1_tag.fits', 'observation2_tag.fits')
    TEXPSTRT f2-f1: -1.5854311641305685e-08 days
    TEXPSTRT f2-f1: -0.0013698125258088112 seconds
    EXPSTART f2-f1: -1.5854311641305685e-08 days
    EXPSTART f2-f1: -0.0013698125258088112 seconds
    First TIME f2-f1: 0.0 seconds
    Last TIME f2-f1: -0.23225000000002183 seconds
    In case it is helpful, the difference between TT and TDB_BJD is -0.001497 s
    """

    # TODO CRH:  it does not appear that any primary header's are being updated with this function call
    with fits.open(file1) as f1, fits.open(file2) as f2:
        # IF STIS
        # add delaytime to TEXPSTRT and TEXPEND, and update primary header
        # COS has no TEXPSTRT in primary header
        if f1[0].header['INSTRUME'] == "STIS":
            print(f"TEXPSTRT file2-file1 {(f2[0].header['TEXPSTRT'] - f1[0].header['TEXPSTRT']) * SECPERDAY} seconds")
            t = Time(f1[0].header['TEXPSTRT'], format='mjd', scale='utc')
        elif f1[0].header['INSTRUME'] == "COS":
            t = Time(f1[1].header['EXPSTART'], format='mjd', scale='utc')
        else:
            raise ValueError(f"Unexpected INSTRUME value: {f1[1].header['EXPSTART']}")

    diff = (t.tdb.value - t.tt.value) * SECPERDAY
    print(f'In case it is helpful, the difference between TT and TDB_BJD is {diff:.6f} s')
