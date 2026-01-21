"""Ingestion subpackage.

Provides provider-agnostic client interfaces and specific implementations
for fetching weather data.
"""

from .client import WeatherClient, OpenMeteoClient

__all__ = ["WeatherClient", "OpenMeteoClient"]

from typing import List, Dict


def fetch_weather(source: str) -> List[Dict[str, float]]:
	"""Compatibility stub for simple ingestion.

	Parameters
	----------
	source : str
		Human-readable source identifier or path/URL.

	Returns
	-------
	List[Dict[str, float]]
		A list of records; each record contains numeric features.

	Notes
	-----
	This function is provided to keep existing smoke tests working. For real
	ingestion, prefer using a concrete `WeatherClient` implementation.
	"""
	return [{"temp": 20.0, "humidity": 0.5, "pressure": 1013.0}]
