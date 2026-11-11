#!/usr/bin/env python3
"""
Module 105-students
Contient la fonction top_students pour PyMongo
"""


def top_students(mongo_collection):
    """
    Renvoie la liste de tous les étudiants triés par moyenne de score.

    Args:
        mongo_collection: Objet collection PyMongo.

    Returns:
        Une liste d'étudiants incluant la clé averageScore, triés de manière décroissante.
    """
    pipeline = [
        {
            "$project": {
                "name": 1,
                "topics": 1,
                "averageScore": {"$avg": "$topics.score"}
            }
        },
        {
            "$sort": {"averageScore": -1}
        }
    ]
    return list(mongo_collection.aggregate(pipeline))
