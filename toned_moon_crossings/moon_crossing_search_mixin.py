from datetime import datetime, timedelta, timezone
from datetime import tzinfo
from typing import Generator, NamedTuple, Optional, Union

from scipy import optimize

from toned_moon_crossings.utils import datetime_as_utc


class CelestialCrossing(NamedTuple):
    time: datetime
    is_rise: Optional[bool]  # True for rise, False for set


class MoonriseMoonset(NamedTuple):
    """
    A tuple of (possibly-None) moonrise and moonset times.

    Attributes
    ----------
    moonrise : datetime | None
        The time of moonrise, or None if not found.
    moonset : datetime | None
        The time of moonset, or None if not found.
    """
    moonrise: Union[datetime, None]
    moonset: Union[datetime, None]


class MoonCrossingSearchMixin:
    @classmethod
    def find_moon_crossings(
            cls,
            observer_latitude: float,
            observer_longitude: float,
            start_datetime: datetime,
            end_datetime: Optional[datetime] = None,
            output_timezone: tzinfo = timezone.utc,
            coarse_interval: timedelta = timedelta(hours=0.5),
    ) -> Generator[CelestialCrossing, None, None]:
        """
        Find all lunar crossings (moonrises and moonssets) for a given location and time range.
        Parameters
        ----------
        observer_latitude : float
            Latitude of the observer in decimal degrees.
        observer_longitude : float
            Longitude of the observer in decimal degrees.
        start_datetime : datetime
            The start time to search for moonrise/set. Defaults to UTC if no timezone is specified.
        end_datetime : datetime | None
            The end time to search for moonrise/set. If None, defaults to start_datetime + 24 hours.
        output_timezone : timezone
            If specified, convert the output times to this timezone (as an offset in hours from UTC).
            If unspecified, defaults to UTC.
        coarse_interval : timedelta
            The time interval for coarse sampling. Default is 0.5 hours.
        Yields
        ------
        CelestialCrossing
            A NamedTuple (time, is_rise) where time is a datetime object and is_rise is True for moonrise,
            False for moonset, or None if undetermined.
            Times are in UTC unless output_timezone is specified.
        """
        start_datetime = datetime_as_utc(start_datetime)

        if end_datetime is None:
            end_datetime = start_datetime + timedelta(hours=24)

        # Sample the 24-hour period in coarse_interval increments to find sign changes
        times = [start_datetime + i * coarse_interval for i in range(0, 49)]  # 0 to 24 in 0.5h steps
        times_as_floats = [t.timestamp() for t in times]  # Convert to timestamps for calculations

        def get_altitude_from_time(t_as_float: float) -> float:
            t_as_datetime = datetime.fromtimestamp(t_as_float, tz=timezone.utc)
            return cls.calc_moon_altitudes(  # type: ignore
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
                yield CelestialCrossing(time=times[i].astimezone(output_timezone), is_rise=is_rise)

            elif values[i] * values[i+1] < 0:
                # Sign change detected, find root in [times[i], times[i+1]]
                lun = optimize.root_scalar(
                    get_altitude_from_time,
                    bracket=[times_as_floats[i], times_as_floats[i+1]],
                    method='brentq'
                )
                if lun.converged:
                    if values[i] < values[i+1]:
                        is_rise = True
                    elif values[i] > values[i+1]:
                        is_rise = False
                    else:
                        raise ValueError("Unexpected condition: values[i] == values[i+1] during root finding.")

                    lun_root_as_datetime = datetime.fromtimestamp(lun.root, tz=timezone.utc)
                    yield CelestialCrossing(time=lun_root_as_datetime.astimezone(output_timezone), is_rise=is_rise)

    @classmethod
    def find_moonrise_moonset(
            cls,
            observer_latitude: float,
            observer_longitude: float,
            start_datetime: datetime,
            end_datetime: Optional[datetime] = None,
            output_timezone: tzinfo = timezone.utc,
            coarse_interval: timedelta = timedelta(hours=0.5),
    ) -> MoonriseMoonset:
        """
        Find the moonrise and moonset times for a given location and start time.
        Parameters
        ----------
        observer_latitude : float
            Latitude of the observer in decimal degrees.
        observer_longitude : float
            Longitude of the observer in decimal degrees.
        start_datetime : datetime
            The start time to search for moonrise/set. Defaults to UTC if no timezone is specified.
        end_datetime : datetime | None
            The end time to search for moonrise/set. If None, defaults to start_datetime + 24 hours.
        output_timezone : float | None
            If specified, convert the output times to this timezone (as an offset in hours from UTC).
        coarse_interval : timedelta
            The time interval for coarse sampling. Default is 0.5 hours.
        Returns
        -------
        DailyMoonCrossings
            A NamedTuple (moonrise, moonset) where each is a datetime object or None if not found.
            Both times are in UTC unless output_timezone is specified.
        """
        start_datetime = datetime_as_utc(start_datetime)

        if end_datetime is None:
            end_datetime = start_datetime + timedelta(hours=24)
        moonrise = None
        moonset = None

        # Get the first crossing time
        crossings_generator = cls.find_moon_crossings(
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
                moonrise = first_crossing.time
            else:
                moonset = first_crossing.time

            if second_crossing and second_crossing.is_rise is not None:
                if second_crossing.is_rise != first_crossing.is_rise:
                    if second_crossing.is_rise:
                        moonrise = second_crossing.time
                    else:
                        moonset = second_crossing.time

        return MoonriseMoonset(moonrise=moonrise, moonset=moonset)

    @classmethod
    def graph_moon_crossings(
            cls,
            observer_latitude: float,
            observer_longitude: float,
            start_datetime: datetime,
            end_datetime: datetime = None,
            output_timezone: tzinfo = timezone.utc,
            coarse_interval: timedelta = timedelta(hours=0.5),
    ) -> None:
        """
        Generate a graph of moonrise and moonset times for a given location and time range.
        Parameters
        ----------
        observer_latitude : float
            Latitude of the observer in decimal degrees.
        observer_longitude : float
            Longitude of the observer in decimal degrees.
        start_datetime : datetime
            The start time to search for moonrise/set. Defaults to UTC if no timezone is specified.
        end_datetime : datetime | None
            The end time to search for moonrise/set. If None, defaults to start_datetime + 24 hours.
        output_timezone : timezone
            If specified, convert the output times to this timezone (as an offset in hours from UTC).
        coarse_interval : timedelta
            The time interval for coarse sampling. Default is 0.5 hours.
        """
        import matplotlib.pyplot as plt

        crossings = list(cls.find_moon_crossings(
            observer_latitude=observer_latitude,
            observer_longitude=observer_longitude,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            output_timezone=output_timezone,
            coarse_interval=coarse_interval,
        ))

        times = [crossing.time for crossing in crossings if crossing.time]
        is_rises = [crossing.is_rise for crossing in crossings if crossing.is_rise is not None]

        plt.figure(figsize=(12, 6))
        plt.plot(times, is_rises, marker='o', linestyle='-', color='blue')
        plt.title('Moonrise and Moonset Times')
        plt.xlabel('Time')
        plt.ylabel('Moonrise (1) / Moonset (0)')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.show()
