#!/usr/bin/env python3
"""
Module fournissant une fonction pour obtenir la liste des vaisseaux
capables de transporter un nombre spécifique de passagers.
"""
import requests


def availableShips(passengerCount):
    """
    Retourne la liste de tous les vaisseaux spatiaux pouvant accueillir
    au moins un nombre donne de passagers.

    Args:
        passengerCount (int): Le nombre minimum de passagers requis.

    Returns:
        list: Une liste contenant les noms des vaisseaux éligibles.
              Retourne une liste vide si aucun vaisseau n'est trouvé.
    """
    url = "https://swapi-api.hbtn.io/api/starships/"
    eligible_ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        results = data.get("results", [])

        for ship in results:
            passengers_str = ship.get("passengers", "0").replace(",", "")
            try:
                passengers_num = int(passengers_str)
                if passengers_num >= passengerCount:
                    eligible_ships.append(ship.get("name"))
            except ValueError:
                # Ignore les valeurs non numériques comme 'unknown' ou 'n/a'
                continue

        url = data.get("next")

    return eligible_ships
