from datetime import datetime
from functools import lru_cache
from typing import Union

import numpy as np
from typing_extensions import override

import astropy
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
import astropy.time
import astropy.units as u
from toned_moon_crossings.sun_calculator_base import SunAltitudes, SunInfoCalculatorBase
from toned_moon_crossings.sun_crossing_search_mixin import SunCrossingSearchMixin


class SunCalculatorAstropy(SunCrossingSearchMixin, SunInfoCalculatorBase):
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_sun_geo(
        date_time: Union[datetime, str],
        observer_latitude: float,
        observer_longitude: float,
        observer_elevation: float = 0.0,
    ) -> SkyCoord:
        ap_time = astropy.time.Time(date_time)
        location = EarthLocation(
            lat=observer_latitude * u.deg,
            lon=observer_longitude * u.deg,
            height=observer_elevation * u.m
        )
        return get_body("sun", ap_time, location=location)

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_altaz_frame(
        date_time: Union[datetime, str],
        observer_latitude: float,
        observer_longitude: float,
        observer_elevation: float = 0.0,
        atmospheric_pressure: float = 1000.0,
    ) -> AltAz:
        ap_time = astropy.time.Time(date_time)
        location = EarthLocation(
            lat=observer_latitude * u.deg,
            lon=observer_longitude * u.deg,
            height=observer_elevation * u.m
        )
        return AltAz(
            obstime=ap_time,
            location=location,
            pressure=atmospheric_pressure * u.hPa
        )

    @staticmethod
    @override
    def calc_sun_altitudes(
        date_time: Union[datetime, str],
        observer_latitude: float,
        observer_longitude: float,
        observer_elevation: float = 0.0,
        atmospheric_pressure: float = 1000.0,
    ) -> SunAltitudes:
        ap_time = astropy.time.Time(date_time)
        location = EarthLocation(
            lat=observer_latitude * u.deg,
            lon=observer_longitude * u.deg,
            height=observer_elevation * u.m
        )
        altaz_frame = AltAz(
            obstime=ap_time,
            location=location,
            pressure=atmospheric_pressure * u.hPa
        )
        sun_geo = SunCalculatorAstropy._get_sun_geo(
            date_time, observer_latitude, observer_longitude,
            observer_elevation
        )

        altaz_frame = SunCalculatorAstropy._get_altaz_frame(
            date_time, observer_latitude, observer_longitude,
            observer_elevation, atmospheric_pressure
        )

        sun_altaz = sun_geo.transform_to(altaz_frame)
        alt_deg = sun_altaz.alt.to_value(u.deg)
        R_sun = 695_700 * u.km
        dist = sun_geo.distance.to(u.km)
        ratio = (R_sun / dist).decompose().value
        ang_radius_deg = np.degrees(np.arcsin(ratio))
        upper_alt = alt_deg + ang_radius_deg
        lower_alt = alt_deg - ang_radius_deg
        return SunAltitudes(
            uppermost_altitude=float(upper_alt),
            lowermost_altitude=float(lower_alt),
        )


if __name__ == "__main__":
    from zoneinfo import ZoneInfo
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from datetime import datetime, timedelta

    latitude = 53.5461  # Edmonton
    longitude = -113.4938
    step = timedelta(minutes=1)
    duration = timedelta(days=1)
    start_dt = datetime.today().astimezone(
        ZoneInfo("America/Edmonton")
    ).replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = start_dt + duration
    pressure_hpa = 950.0

    times = []
    ap_upper, ap_lower = [], []

    t = start_dt
    while t < end_dt:
        upper, lower = SunCalculatorAstropy.calc_sun_altitudes(
            t, latitude, longitude, atmospheric_pressure=pressure_hpa
        )
        times.append(t)
        ap_upper.append(upper)
        ap_lower.append(lower)
        t += step

    crossings = SunCalculatorAstropy.find_sun_crossings(
        observer_latitude=latitude,
        observer_longitude=longitude,
        start_datetime=start_dt,
        output_timezone=ZoneInfo("America/Edmonton"),
    )

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(
        nrows=2, ncols=1, height_ratios=[1, 0.6], hspace=0.22
    )

    ax_top = fig.add_subplot(gs[0])
    ax_cross = fig.add_subplot(gs[1], sharex=ax_top)

    print(times[0])
    edmonton_time = ZoneInfo("America/Edmonton")
    times = [t.astimezone(edmonton_time) for t in times]  # Convert to local tz
    print(times[0])

    # Top: Astropy
    ax_top.plot(times, ap_upper, label="upper")
    ax_top.plot(times, ap_lower, label="lower")
    ax_top.set_ylabel("Altitude (deg)")
    ax_top.set_title("AstroPy solar upper/lower altitudes")
    ax_top.legend(loc="upper left")

    sunrise_times, sunset_times = [], []
    for crossing in crossings:
        local_time = crossing.time
        if crossing.is_rise:
            sunrise_times.append(local_time)
        elif crossing.is_rise is False:
            sunset_times.append(local_time)

    ax_cross.vlines(
        sunrise_times, ymin=min(ap_lower), ymax=max(ap_upper), colors="orange", linestyle="--"
    )
    ax_cross.vlines(
        sunset_times, ymin=min(ap_lower), ymax=max(ap_upper), colors="purple", linestyle=":"
    )
    ax_cross.set_ylabel("Altitude (deg)")
    ax_cross.set_title("Sunrise/Sunset crossings")

    legend_elements = [
        Line2D([0], [0], color="orange", lw=2, linestyle="--", label="Sunrise"),
        Line2D([0], [0], color="purple", lw=2, linestyle=":", label="Sunset"),
    ]
    ax_cross.legend(handles=legend_elements, loc="upper left")

    plt.show()
