"""unit_converter — improved converter toolset (demo target for editing).

Improvements over the starter:
  * Input validation with clear error dicts instead of raw exceptions.
  * Optional rounding for all conversions.
  * Reverse conversions (F->C, miles->km).
  * A generic ``tool_convert`` dispatcher.
"""

from typing import Any, Dict, Optional

TOOL_LIBRARY_NAME = "Unit Converter"
TOOL_LIBRARY_DESC = "Convert values between units (demo target for editing)."
TOOL_LIBRARY_ICON = "🔄"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _error(message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message}


def _coerce_number(value: Any, name: str) -> Optional[float]:
    """Return ``value`` as float, or None if it is not a finite number."""
    if isinstance(value, bool):  # bool is an int subclass — reject it
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num or num in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return num


def _round(value: float, digits: Optional[int]) -> float:
    if digits is None:
        return value
    return round(value, int(digits))


def init_tools_library(config: dict = None) -> None:
    return None


# ---------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------


def tool_celsius_to_fahrenheit(celsius: float, round_digits: int = None) -> Dict[str, Any]:
    """Convert a temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in degrees Celsius.
        round_digits: Optional number of decimal places to round the result to.
    """
    c = _coerce_number(celsius, "celsius")
    if c is None:
        return _error(f"celsius must be a finite number, got {celsius!r}")
    f = c * 9.0 / 5.0 + 32.0
    return {
        "ok": True,
        "celsius": c,
        "fahrenheit": _round(f, round_digits),
    }


def tool_fahrenheit_to_celsius(fahrenheit: float, round_digits: int = None) -> Dict[str, Any]:
    """Convert a temperature from Fahrenheit to Celsius.

    Args:
        fahrenheit: Temperature in degrees Fahrenheit.
        round_digits: Optional number of decimal places to round the result to.
    """
    f = _coerce_number(fahrenheit, "fahrenheit")
    if f is None:
        return _error(f"fahrenheit must be a finite number, got {fahrenheit!r}")
    c = (f - 32.0) * 5.0 / 9.0
    return {
        "ok": True,
        "fahrenheit": f,
        "celsius": _round(c, round_digits),
    }


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------

_KM_PER_MILE = 1.609344


def tool_km_to_miles(kilometers: float, round_digits: int = None) -> Dict[str, Any]:
    """Convert a distance from kilometers to miles.

    Args:
        kilometers: Distance in kilometers.
        round_digits: Optional number of decimal places to round the result to.
    """
    km = _coerce_number(kilometers, "kilometers")
    if km is None:
        return _error(f"kilometers must be a finite number, got {kilometers!r}")
    miles = km / _KM_PER_MILE
    return {
        "ok": True,
        "kilometers": km,
        "miles": _round(miles, round_digits),
    }


def tool_miles_to_km(miles: float, round_digits: int = None) -> Dict[str, Any]:
    """Convert a distance from miles to kilometers.

    Args:
        miles: Distance in miles.
        round_digits: Optional number of decimal places to round the result to.
    """
    mi = _coerce_number(miles, "miles")
    if mi is None:
        return _error(f"miles must be a finite number, got {miles!r}")
    km = mi * _KM_PER_MILE
    return {
        "ok": True,
        "miles": mi,
        "kilometers": _round(km, round_digits),
    }


# ---------------------------------------------------------------------------
# generic dispatcher
# ---------------------------------------------------------------------------

# Canonical unit aliases -> canonical unit name.
_UNIT_ALIASES = {
    "c": "celsius",
    "celsius": "celsius",
    "°c": "celsius",
    "degc": "celsius",
    "f": "fahrenheit",
    "fahrenheit": "fahrenheit",
    "°f": "fahrenheit",
    "degf": "fahrenheit",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "mi": "miles",
    "mile": "miles",
    "miles": "miles",
}


def _canonical_unit(unit: Any) -> Optional[str]:
    """Map a user-supplied unit string to its canonical name."""
    if unit is None:
        return None
    return _UNIT_ALIASES.get(str(unit).strip().lower())


# canonical pair -> (conversion fn, source result key, target result key)
_CONVERSIONS = {
    ("celsius", "fahrenheit"): (tool_celsius_to_fahrenheit, "celsius", "fahrenheit"),
    ("fahrenheit", "celsius"): (tool_fahrenheit_to_celsius, "fahrenheit", "celsius"),
    ("km", "miles"): (tool_km_to_miles, "kilometers", "miles"),
    ("miles", "km"): (tool_miles_to_km, "miles", "kilometers"),
}


def tool_convert(value: float, from_unit: str, to_unit: str, round_digits: int = None) -> Dict[str, Any]:
    """Convert ``value`` between supported units.

    Supported unit pairs: celsius<->fahrenheit, km<->miles.

    Args:
        value: The numeric value to convert.
        from_unit: Source unit ('celsius', 'fahrenheit', 'km', 'miles').
        to_unit: Target unit ('celsius', 'fahrenheit', 'km', 'miles').
        round_digits: Optional number of decimal places to round the result to.
    """
    if from_unit is None or to_unit is None:
        return _error("from_unit and to_unit are required")

    src = _canonical_unit(from_unit)
    dst = _canonical_unit(to_unit)
    if src is None:
        return _error(
            f"unknown unit {from_unit!r}; supported: {sorted(set(_UNIT_ALIASES.values()))}"
        )
    if dst is None:
        return _error(
            f"unknown unit {to_unit!r}; supported: {sorted(set(_UNIT_ALIASES.values()))}"
        )
    if src == dst:
        num = _coerce_number(value, "value")
        if num is None:
            return _error(f"value must be a finite number, got {value!r}")
        return {
            "ok": True,
            "input": {"value": num, "unit": src},
            "output": {"unit": dst, "value": _round(num, round_digits)},
            "detail": {src: num},
        }

    entry = _CONVERSIONS.get((src, dst))
    if entry is None:
        return _error(
            f"unsupported conversion {src!r} -> {dst!r}; "
            f"supported pairs: {sorted(_CONVERSIONS)}"
        )
    fn, _src_key, dst_key = entry

    result = fn(value, round_digits=round_digits)
    if not result.get("ok"):
        return result
    return {
        "ok": True,
        "input": {"value": result.get(_src_key), "unit": src},
        "output": {"unit": dst, "value": result.get(dst_key)},
        "detail": result,
    }
