#!/usr/bin/env python3
"""
Module fournissant une fonction pour obtenir la liste des planètes
d'origine de toutes les espèces douées de conscience.
"""
import requests


def sentientPlanets():
    """
    Retourne la liste des noms des planètes d'origine
    de toutes les espèces douées de conscience (sentient).

    Returns:
        list: Une liste de chaînes de caractères
        contenant les noms des planètes.
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        results = data.get("results", [])

        for species in results:
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")
                if homeworld_url:
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_name = planet_response.json().get("name")
                        if planet_name:
                            planets.append(planet_name)

        url = data.get("next")

    return planets
