"""
Tunisia geographic data for mesh network simulation
Uses real city locations and terrain types
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class TunisianCity:
    """Represents a Tunisian city/region"""
    name: str
    x: float  # Simplified coordinates (0-1000 scale)
    y: float
    population: int
    terrain_type: str  # 'coastal', 'urban', 'desert', 'mountain'
    elevation: float  # meters above sea level
    
    def __repr__(self):
        return f"{self.name} ({self.terrain_type})"


# Real Tunisian cities with simplified coordinates
# x represents East-West (0=West, 1000=East)
# y represents North-South (0=North, 1000=South)
TUNISIAN_CITIES = {
    # Northern coastal cities
    'tunis': TunisianCity('Tunis', 500, 150, 1056247, 'urban', 4),
    'ariana': TunisianCity('Ariana', 480, 140, 97687, 'urban', 20),
    'bizerte': TunisianCity('Bizerte', 450, 100, 142966, 'coastal', 3),
    'nabeul': TunisianCity('Nabeul', 550, 180, 73128, 'coastal', 5),
    'hammamet': TunisianCity('Hammamet', 560, 200, 73236, 'coastal', 2),
    
    # Central coastal cities
    'sousse': TunisianCity('Sousse', 520, 400, 221530, 'coastal', 4),
    'monastir': TunisianCity('Monastir', 530, 420, 93306, 'coastal', 3),
    'mahdia': TunisianCity('Mahdia', 540, 460, 62189, 'coastal', 2),
    'sfax': TunisianCity('Sfax', 550, 600, 277278, 'coastal', 2),
    
    # Interior cities
    'kairouan': TunisianCity('Kairouan', 480, 450, 139070, 'urban', 67),
    'kasserine': TunisianCity('Kasserine', 400, 550, 76243, 'mountain', 670),
    'sidi_bouzid': TunisianCity('Sidi Bouzid', 450, 580, 42098, 'urban', 346),
    'gafsa': TunisianCity('Gafsa', 350, 650, 95242, 'desert', 307),
    
    # Southern cities
    'gabes': TunisianCity('Gabès', 520, 700, 130984, 'coastal', 4),
    'medenine': TunisianCity('Médenine', 580, 800, 61705, 'desert', 62),
    'tataouine': TunisianCity('Tataouine', 600, 850, 59346, 'desert', 206),
    'tozeur': TunisianCity('Tozeur', 300, 720, 38889, 'desert', 85),
    'kebili': TunisianCity('Kébili', 350, 770, 47908, 'desert', 38),
    
    # Western mountain regions
    'jendouba': TunisianCity('Jendouba', 350, 250, 51408, 'mountain', 143),
    'le_kef': TunisianCity('Le Kef', 380, 280, 45191, 'mountain', 780),
}


# Terrain characteristics
TERRAIN_TYPES = {
    'coastal': {
        'description': 'Coastal areas prone to flooding',
        'flood_risk': 0.9,  # 0-1 scale
        'earthquake_risk': 0.2,
        'signal_quality': 0.8,  # Better connectivity
    },
    'urban': {
        'description': 'Dense urban areas with infrastructure',
        'flood_risk': 0.5,
        'earthquake_risk': 0.4,
        'signal_quality': 0.9,
    },
    'mountain': {
        'description': 'Mountainous regions with difficult terrain',
        'flood_risk': 0.2,
        'earthquake_risk': 0.7,
        'signal_quality': 0.4,  # Poor connectivity
    },
    'desert': {
        'description': 'Desert and arid regions',
        'flood_risk': 0.1,
        'earthquake_risk': 0.3,
        'signal_quality': 0.3,  # Sparse infrastructure
    }
}


def get_cities_by_terrain(terrain_type: str) -> List[TunisianCity]:
    """Get all cities of a specific terrain type"""
    return [city for city in TUNISIAN_CITIES.values() 
            if city.terrain_type == terrain_type]


def get_cities_in_region(x_min: float, x_max: float, 
                         y_min: float, y_max: float) -> List[TunisianCity]:
    """Get cities within a bounding box"""
    return [city for city in TUNISIAN_CITIES.values()
            if x_min <= city.x <= x_max and y_min <= city.y <= y_max]


def get_coastal_cities() -> List[TunisianCity]:
    """Get all coastal cities (high flood risk)"""
    return get_cities_by_terrain('coastal')


def get_desert_cities() -> List[TunisianCity]:
    """Get all desert cities (isolated areas)"""
    return get_cities_by_terrain('desert')


def calculate_distance(city1: TunisianCity, city2: TunisianCity) -> float:
    """Calculate simplified distance between two cities"""
    return ((city1.x - city2.x)**2 + (city1.y - city2.y)**2)**0.5


# Example disaster-prone zones
DISASTER_ZONES = {
    'nabeul_flood_zone': {
        'cities': ['nabeul', 'hammamet'],
        'disaster_type': 'flooding',
        'description': 'Coastal flooding in Cap Bon region',
        'frequency': 'high'
    },
    'bizerte_coastal': {
        'cities': ['bizerte'],
        'disaster_type': 'flooding',
        'description': 'Northern coastal flooding',
        'frequency': 'medium'
    },
    'tunis_infrastructure': {
        'cities': ['tunis', 'ariana'],
        'disaster_type': 'infrastructure_failure',
        'description': 'Urban infrastructure overload',
        'frequency': 'medium'
    },
    'kasserine_earthquake': {
        'cities': ['kasserine', 'le_kef'],
        'disaster_type': 'earthquake',
        'description': 'Mountainous seismic activity',
        'frequency': 'low'
    },
    'desert_isolation': {
        'cities': ['tozeur', 'kebili', 'tataouine'],
        'disaster_type': 'communication_loss',
        'description': 'Remote area connectivity loss',
        'frequency': 'medium'
    }
}


def get_disaster_zone_cities(zone_name: str) -> List[TunisianCity]:
    """Get cities in a specific disaster zone"""
    if zone_name not in DISASTER_ZONES:
        return []
    city_names = DISASTER_ZONES[zone_name]['cities']
    return [TUNISIAN_CITIES[name] for name in city_names if name in TUNISIAN_CITIES]