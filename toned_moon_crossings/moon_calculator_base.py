from abc import ABC, abstractmethod
from datetime import datetime
from typing import NamedTuple, Union


class MoonAltitudes(NamedTuple):
    """
    A named tuple to hold the moon's altitude information.
    Attributes
    ----------
    uppermost_altitude : float
        The highest altitude of the moon in degrees.
    lowermost_altitude : float
        The lowest altitude of the moon in degrees.
    """
    uppermost_altitude: float
    lowermost_altitude: float


class MoonInfoCalculatorBase(ABC):
    @staticmethod
    @abstractmethod
    def calc_moon_altitudes(
        date_time: Union[datetime, str],
        observer_latitude: float,
        observer_longitude: float,
        observer_elevation: float = 0.0,
        atmospheric_pressure: float = 1000.0,
    ) -> MoonAltitudes:
        """
        Calculate if the moon is visible based on its altitude and azimuth.
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

    @staticmethod
    @abstractmethod
    def calc_moon_phase_proportion(
        date_time: Union[datetime, str],
        observer_latitude: float,
        observer_longitude: float,
        observer_elevation: float = 0.0,
        atmospheric_pressure: float = 1000.0,
    ) -> float:
        """
        Calculate the proportion of the (Earth-facing side of the) moon that's illuminated.

        Parameters
        ----------
        date_time : datetime | str
            The date and time for which to calculate the moon phase.
                If a string is provided, it should be in ISO 8601 format
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
