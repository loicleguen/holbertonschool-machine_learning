#!/usr/bin/env python3
"""
Module 30-all
Contient la fonction list_all pour PyMongo
"""


def list_all(mongo_collection):
    """
    Liste tous les documents d'une collection MongoDB.

    Args:
        mongo_collection: Objet collection PyMongo.

    Returns:
        Une liste de tous les documents ou une liste vide s'il n'y en a aucun.
    """
    return list(mongo_collection.find())
