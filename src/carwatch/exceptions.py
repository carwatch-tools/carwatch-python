"""Exceptions raised by :mod:`carwatch`."""


class CarwatchError(Exception):
    """Base class for CARWatch errors."""


class LogParseError(CarwatchError, ValueError):
    """Raised when a CARWatch log entry cannot be parsed."""


class SchemaError(CarwatchError, ValueError):
    """Raised when input data do not follow the expected schema."""
