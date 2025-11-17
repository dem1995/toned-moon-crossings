from abc import ABC, abstractmethod
from datetime import datetime
from typing import NamedTuple, Union


class SunAltitudes(NamedTuple):
    """
    A named tuple to hold the sun's altitude information.

    Attributes
    ----------
    uppermost_altitude : float
        The highest altitude of the sun in degrees.
    lowermost_altitude : float
        The lowest altitude of the sun in degrees.
    """

    uppermost_altitude: float
    lowermost_altitude: float


class SunInfoCalculatorBase(ABC):
    @staticmethod
    @abstractmethod
    def calc_sun_altitudes(
        date_time: Union[datetime, str],
        observer_latitude: float,
        observer_longitude: float,
        observer_elevation: float = 0.0,
        atmospheric_pressure: float = 1000.0,
    ) -> SunAltitudes:
        """
        Calculate if the sun is visible based on its altitude and azimuth.

        Parameters
        ----------
        date_time : datetime | str
            The date and time for which to calculate visibility.
                If a string is provided, it should be in ISO 8601 format.
                    (e.g., "2023-10-01T12:00:00").
        observer_latitude : float
            The latitude of the observer in degrees.
        observer_longitude : float
            The longitude of the observer in degrees.
        observer_elevation : float
            The elevation of the observer in meters.
        atmospheric_pressure : float
            The atmospheric pressure in hPa.
        """

        pass
