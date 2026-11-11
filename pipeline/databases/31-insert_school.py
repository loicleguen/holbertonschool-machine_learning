#!/usr/bin/env python3
"""
Module 31-insert_school
Contient la fonction insert_school pour PyMongo
"""


def insert_school(mongo_collection, **kwargs):
    """
    Insère un nouveau document dans une collection basé sur kwargs.

    Args:
        mongo_collection: Objet collection PyMongo.
        **kwargs: Attributs clé/valeur du document à insérer.

    Returns:
        L'ID (_id) du nouveau document créé.
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
