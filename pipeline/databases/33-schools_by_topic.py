#!/usr/bin/env python3
"""
Module 33-schools_by_topic
Contient la fonction schools_by_topic pour PyMongo
"""


def schools_by_topic(mongo_collection, topic):
    """
    Renvoie la liste des écoles ayant un topic spécifique.

    Args:
        mongo_collection: Objet collection PyMongo.
        topic (str): Le topic recherché dans la liste des topics de chaque école.

    Returns:
        Une liste de dictionnaires représentant les écoles correspondantes.
    """
    return list(mongo_collection.find({"topics": topic}))
