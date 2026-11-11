#!/usr/bin/env python3
"""
Module 32-update_topics
Contient la fonction update_topics pour PyMongo
"""


def update_topics(mongo_collection, name, topics):
    """
    Met à jour tous les sujets (topics) d'une école selon son nom.

    Args:
        mongo_collection: Objet collection PyMongo.
        name (str): Nom de l'école à mettre à jour.
        topics (list of str): Liste des sujets à attribuer.
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
