"""
Disaster scenarios and Tunisia geographic data.
"""
from src.scenarios.tunisia_map import (
    TUNISIAN_CITIES, TunisianCity, TERRAIN_TYPES,
    get_coastal_cities, get_desert_cities,
)
from src.scenarios.disaster import (
    FloodingScenario, EarthquakeScenario, InfrastructureFailureScenario,
    create_nabeul_flood, create_tunis_infrastructure_collapse, create_kasserine_earthquake,
)

__all__ = [
    "TUNISIAN_CITIES", "TunisianCity", "TERRAIN_TYPES",
    "get_coastal_cities", "get_desert_cities",
    "FloodingScenario", "EarthquakeScenario", "InfrastructureFailureScenario",
    "create_nabeul_flood", "create_tunis_infrastructure_collapse", "create_kasserine_earthquake",
]
