from datetime import datetime, timezone
from typing import Optional


def datetime_as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Supplies the datetime in UTC.
    If the datetime is None, returns None.
    If the datetime has no timezone, sets the timezone to UTC.
    If the datetime is in a different timezone, converts it to UTC.

    Parameters
    ----------
    dt : Optional[datetime]
        The datetime to make use UTC.
    Returns
    -------
    Optional[datetime]
        The datetime in UTC.
    """

    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
