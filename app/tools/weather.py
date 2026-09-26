"""Weather tool: current weather for a place name, via the free Open-Meteo API.

Open-Meteo needs no API key, which keeps this project runnable by anyone who
clones it. Two HTTP calls are made:
  1. Geocoding API  - turn a place name ("Dublin") into latitude/longitude.
  2. Forecast API   - turn latitude/longitude into current weather.
"""

import requests

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes -> human-readable text (Open-Meteo returns a numeric code).
_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Freezing fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Light rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


def get_weather(place: str, timeout: float = 10.0) -> str:
    """Return a one-line current weather summary for `place`."""
    place = place.strip()
    if not place:
        raise ValueError("Place name must not be empty.")

    geo_resp = requests.get(
        _GEOCODE_URL, params={"name": place, "count": 1}, timeout=timeout
    )
    geo_resp.raise_for_status()
    geo_results = geo_resp.json().get("results")
    if not geo_results:
        raise ValueError(f"Could not find a location named '{place}'.")

    location = geo_results[0]
    lat, lon = location["latitude"], location["longitude"]
    resolved_name = f"{location['name']}, {location.get('country', '')}".strip(", ")

    forecast_resp = requests.get(
        _FORECAST_URL,
        params={"latitude": lat, "longitude": lon, "current_weather": True},
        timeout=timeout,
    )
    forecast_resp.raise_for_status()
    current = forecast_resp.json().get("current_weather")
    if not current:
        raise ValueError(f"No current weather data available for '{place}'.")

    condition = _WEATHER_CODES.get(current["weathercode"], "Unknown conditions")
    return (
        f"Current weather in {resolved_name}: {current['temperature']}°C, "
        f"wind {current['windspeed']} km/h, {condition}."
    )


TOOL_SPEC = {
    "name": "weather",
    "description": (
        "Get the current weather for a place. Input is just a place name, "
        "e.g. 'Tokyo' or 'Dublin, Ireland'."
    ),
}
