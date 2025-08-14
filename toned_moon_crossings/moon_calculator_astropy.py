from datetime import datetime
from functools import lru_cache
from typing import Union

import numpy as np
from typing_extensions import override

import astropy
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
import astropy.time
import astropy.units as u
from toned_moon_crossings.moon_calculator_base import (
    MoonAltitudes,
    MoonInfoCalculatorBase,
)
from toned_moon_crossings.moon_crossing_search_mixin import MoonCrossingSearchMixin


class MoonCalculatorAstropy(MoonCrossingSearchMixin, MoonInfoCalculatorBase):
    @staticmethod
    @lru_cache(maxsize=128)
    def _get_moon_geo(
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
        return get_body("moon", ap_time, location=location)

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
    def calc_moon_altitudes(
        date_time: Union[datetime, str],
        observer_latitude: float,
        observer_longitude: float,
        observer_elevation: float = 0.0,
        atmospheric_pressure: float = 1000.0,
    ) -> MoonAltitudes:
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
        moon_geo = MoonCalculatorAstropy._get_moon_geo(
            date_time, observer_latitude, observer_longitude,
            observer_elevation
        )

        altaz_frame = MoonCalculatorAstropy._get_altaz_frame(
            date_time, observer_latitude, observer_longitude,
            observer_elevation, atmospheric_pressure
        )

        moon_altaz = moon_geo.transform_to(altaz_frame)
        alt_deg = moon_altaz.alt.to_value(u.deg)
        R_moon = 1737.4 * u.km
        dist = moon_geo.distance.to(u.km)
        ratio = (R_moon / dist).decompose().value  # dimensionless float
        ang_radius_deg = np.degrees(np.arcsin(ratio))  # plain float in degrees
        upper_alt = alt_deg + ang_radius_deg
        lower_alt = alt_deg - ang_radius_deg
        return MoonAltitudes(
            uppermost_altitude=float(upper_alt),
            lowermost_altitude=float(lower_alt),
        )

    @staticmethod
    @override
    def calc_moon_phase_proportion(
        date_time: Union[datetime, str],
        observer_latitude: float,
        observer_longitude: float,
        observer_elevation: float = 0.0,
        atmospheric_pressure: float = 1000.0,
    ) -> float:
        ap_time = astropy.time.Time(date_time)
        location = EarthLocation(
            lat=observer_latitude * u.deg,
            lon=observer_longitude * u.deg,
            height=observer_elevation * u.m
        )
        moon_geo = MoonCalculatorAstropy._get_moon_geo(
            date_time, observer_latitude, observer_longitude,
            observer_elevation
        )
        sun_geo = get_body("sun", ap_time, location=location)
        epsilon = sun_geo.separation(moon_geo).to_value(u.rad)
        phase_frac = 0.5 * (1.0 - np.cos(epsilon))
        return float(phase_frac)


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
    ap_upper, ap_lower, ap_phase = [], [], []
    # ep_upper, ep_lower, ep_phase = [], [], []

    t = start_dt
    while t < end_dt:
        upper, lower = MoonCalculatorAstropy.calc_moon_altitudes(
            t, latitude, longitude, atmospheric_pressure=pressure_hpa
        )
        phase = MoonCalculatorAstropy.calc_moon_phase_proportion(
            t, latitude, longitude, atmospheric_pressure=pressure_hpa
        )
        times.append(t)
        ap_upper.append(upper)
        ap_lower.append(lower)
        ap_phase.append(phase)
        t += step

    crossings = MoonCalculatorAstropy.find_moon_crossings(
        observer_latitude=latitude,
        observer_longitude=longitude,
        start_datetime=start_dt,
        output_timezone=ZoneInfo("America/Edmonton"),
    )

    def pct_diff(a, b):
        denom = (abs(a) + abs(b)) / 2.0
        return 0.0 if denom == 0 else 100.0 * (a - b) / denom

    pct_upper = [pct_diff(a, b) for a, b in zip(ap_upper, ap_upper)]
    pct_lower = [pct_diff(a, b) for a, b in zip(ap_lower, ap_lower)]

    abs_upper = [abs(a - b) for a, b in zip(ap_upper, ap_upper)]
    abs_lower = [abs(a - b) for a, b in zip(ap_lower, ap_lower)]

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(
        nrows=4, ncols=1, height_ratios=[1, 1, 0.8, 0.8], hspace=0.22
    )

    ax_top = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1], sharex=ax_top)
    ax_pct = fig.add_subplot(gs[2], sharex=ax_top)
    ax_abs = fig.add_subplot(gs[3], sharex=ax_top)

    print(times[0])
    edmonton_time = ZoneInfo("America/Edmonton")
    times = [t.astimezone(edmonton_time) for t in times]  # Convert to local tz
    print(times[0])

    # Top: Astropy
    ax_top.plot(times, ap_upper, label="upper")
    ax_top.plot(times, ap_lower, label="lower")
    ax_top_phase = ax_top.twinx()
    ax_top_phase.plot(times, ap_phase, label="phase", linestyle=":")
    ax_top.set_ylabel("Altitude (deg)")
    ax_top.set_title("AstroPy lunar upper/lower altitudes and phase")
    ax_top.legend(loc="upper left")
    ax_top_phase.legend(loc="upper right")

    # Middle: Ephem
    ax_mid.plot(times, ap_upper, label="upper")
    ax_mid.plot(times, ap_lower, label="lower")
    ax_mid_phase = ax_mid.twinx()
    ax_mid_phase.plot(times, ap_phase, label="phase", linestyle=":")
    ax_mid.set_ylabel("Altitude (deg)")
    ax_mid.set_title("PyEphem lunar upper/lower altitudes and phase")
    ax_mid.legend(loc="upper left")
    ax_mid_phase.legend(loc="upper right")

    # 3rd: % difference
    ax_pct.plot(times, pct_upper, label="% diff upper")
    ax_pct.plot(times, pct_lower, label="% diff lower")
    ax_pct.axhline(0, linewidth=1, color="black")
    ax_pct.set_ylabel("% diff")
    ax_pct.set_title(
        "Symmetric % difference (100*(ap-ephem)/((|ap|+|ephem|)/2))"
    )
    ax_pct.legend(loc="upper left")

    # 4th: absolute difference
    ax_abs.plot(times, abs_upper, label="abs diff upper")
    ax_abs.plot(times, abs_lower, label="abs diff lower")
    ax_abs.axhline(0, linewidth=1, color="black")
    ax_abs.set_ylabel("Abs diff (deg)")
    ax_abs.set_xlabel("Datetime (local)")
    ax_abs.set_title("Absolute difference (degrees)")
    ax_abs.legend(loc="upper left")

    # Horizon crossings + labels
    vline_kwargs = dict(color="red", linestyle="--", linewidth=1)
    crossing_proxy = Line2D(
        [0], [0],
        **vline_kwargs,
        label="Lunar horizon crossing (ap lower)"
    )

    for c in crossings:
        for ax, y_pos in [
            (ax_top, 0),       # label at y=0 in altitude plot
            (ax_mid, 0),       # label at y=0 in altitude plot
            (ax_pct, min(min(pct_upper), min(pct_lower)) - 2),  # just below min % diff
            (ax_abs, min(min(abs_upper), min(abs_lower)) - 0.1)  # just below min abs diff
        ]:
            ax.axvline(c.time, **vline_kwargs)
            ax.text(c.time, y_pos, c.time.strftime('%H:%M'), rotation=45,
                    ha='right', va='bottom', fontsize=8)

    # Add crossing legend to each subplot
    for ax in [ax_top, ax_mid, ax_pct, ax_abs]:
        handles, labels = ax.get_legend_handles_labels()
        if "Lunar horizon crossing (ap lower)" not in labels:
            ax.legend(
                handles + [crossing_proxy],
                labels + ["Lunar horizon crossing (ap lower)"],
                loc="upper left"
            )

    plt.setp(ax_mid.get_xticklabels(), visible=False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_pct.get_xticklabels(), visible=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
