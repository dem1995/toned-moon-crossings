from datetime import datetime, timedelta, timezone
from datetime import tzinfo
from typing import Generator, NamedTuple, Optional

from scipy import optimize


class SolarCrossing(NamedTuple):
    time: datetime
    is_rise: Optional[bool]  # True for rise, False for set


class SunriseSunset(NamedTuple):
    """
    A tuple of (possibly-None) sunrise and sunset times.

    Attributes
    ----------
    sunrise : datetime | None
        The time of sunrise, or None if not found.
    sunset : datetime | None
        The time of sunset, or None if not found.
    """

    sunrise: Optional[datetime]
    sunset: Optional[datetime]


class SunCrossingSearchMixin:
    @classmethod
    def find_sun_crossings(
            cls,
            observer_latitude: float,
            observer_longitude: float,
            start_datetime: datetime,
            end_datetime: Optional[datetime] = None,
            output_timezone: tzinfo = timezone.utc,
            coarse_interval: timedelta = timedelta(hours=0.5),
    ) -> Generator[SolarCrossing, None, None]:
        """
        Find all solar crossings (sunrises and sunsets) for a given location and time range.
        Parameters
        ----------
        observer_latitude : float
            Latitude of the observer in decimal degrees.
        observer_longitude : float
            Longitude of the observer in decimal degrees.
        start_datetime : datetime
            The start time to search for sunrise/set. Defaults to UTC if no timezone is specified.
        end_datetime : datetime | None
            The end time to search for sunrise/set. If None, defaults to start_datetime + 24 hours.
        output_timezone : timezone
            If specified, convert the output times to this timezone (as an offset in hours from UTC).
            If unspecified, defaults to UTC.
        coarse_interval : timedelta
            The time interval for coarse sampling. Default is 0.5 hours.
        Yields
        ------
        SolarCrossing
            A NamedTuple (time, is_rise) where time is a datetime object and is_rise is True for sunrise,
            False for sunset, or None if undetermined.
            Times are in UTC unless output_timezone is specified.
        """

        if end_datetime is None:
            end_datetime = start_datetime + timedelta(hours=24)

        times = [start_datetime + i * coarse_interval for i in range(0, 49)]  # 0 to 24 in 0.5h steps
        times_as_floats = [t.timestamp() for t in times]

        def get_altitude_from_time(t_as_float: float) -> float:
            t_as_datetime = datetime.fromtimestamp(t_as_float)
            return cls.calc_sun_altitudes(  # type: ignore
                date_time=t_as_datetime,
                observer_latitude=observer_latitude,
                observer_longitude=observer_longitude,
            ).uppermost_altitude

        values = [get_altitude_from_time(time_as_float) for time_as_float in times_as_floats]

        for i in range(len(times) - 1):
            if values[i] == 0:
                if i < len(values) - 1:
                    is_rise = values[i+1] > 0
                elif i > 0:
                    is_rise = values[i-1] < 0
                else:
                    is_rise = None
                yield SolarCrossing(time=times[i].astimezone(output_timezone), is_rise=is_rise)

            elif values[i] * values[i+1] < 0:
                sol = optimize.root_scalar(
                    get_altitude_from_time,
                    bracket=[times_as_floats[i], times_as_floats[i+1]],
                    method='brentq'
                )
                if sol.converged:
                    if values[i] < values[i+1]:
                        is_rise = True
                    elif values[i] > values[i+1]:
                        is_rise = False
                    else:
                        raise ValueError("Unexpected condition: values[i] == values[i+1] during root finding.")

                    sol_root_as_datetime = datetime.fromtimestamp(sol.root)
                    yield SolarCrossing(time=sol_root_as_datetime.astimezone(output_timezone), is_rise=is_rise)

    @classmethod
    def find_sunrise_sunset(
            cls,
            observer_latitude: float,
            observer_longitude: float,
            start_datetime: datetime,
            end_datetime: Optional[datetime] = None,
            output_timezone: tzinfo = timezone.utc,
            coarse_interval: timedelta = timedelta(hours=0.5),
    ) -> SunriseSunset:
        """
        Find the sunrise and sunset times for a given location and start time.

        Parameters
        ----------
        observer_latitude : float
            Latitude of the observer in decimal degrees.
        observer_longitude : float
            Longitude of the observer in decimal degrees.
        start_datetime : datetime
            The start time to search for sunrise/set. Defaults to UTC if no timezone is specified.
        end_datetime : datetime | None
            The end time to search for sunrise/set. If None, defaults to start_datetime + 24 hours.
        output_timezone : float | None
            If specified, convert the output times to this timezone (as an offset in hours from UTC).
        coarse_interval : timedelta
            The time interval for coarse sampling. Default is 0.5 hours.
        Returns
        -------
        SunriseSunset
            A NamedTuple (sunrise, sunset) where each is a datetime object or None if not found.
            Both times are in UTC unless output_timezone is specified.
        """

        if start_datetime.tzinfo is None:
            start_datetime = start_datetime.replace(tzinfo=timezone.utc)
        if end_datetime is None:
            end_datetime = start_datetime + timedelta(hours=24)
        sunrise = None
        sunset = None

        crossings_generator = cls.find_sun_crossings(
            observer_latitude=observer_latitude,
            observer_longitude=observer_longitude,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            output_timezone=output_timezone,
            coarse_interval=coarse_interval,
        )

        first_crossing = next(crossings_generator, None)
        second_crossing = next(crossings_generator, None)

        if first_crossing and first_crossing.is_rise is not None:
            if first_crossing.is_rise:
                sunrise = first_crossing.time
            else:
                sunset = first_crossing.time

            if second_crossing and second_crossing.is_rise is not None:
                if second_crossing.is_rise != first_crossing.is_rise:
                    if second_crossing.is_rise:
                        sunrise = second_crossing.time
                    else:
                        sunset = second_crossing.time

        return SunriseSunset(sunrise=sunrise, sunset=sunset)
