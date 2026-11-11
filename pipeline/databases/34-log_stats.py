#!/usr/bin/env python3
"""
Module 34-log_stats
Fournit des statistiques sur les logs Nginx stockés dans MongoDB
"""
from pymongo import MongoClient


def log_stats():
    """
    Affiche les statistiques de la collection logs.nginx:
    - Nombre total de logs
    - Nombre de logs par méthode (GET, POST, PUT, PATCH, DELETE)
    - Nombre de logs avec method=GET et path=/status
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx

    # Total des documents
    total_logs = collection.count_documents({})
    print("{} logs".format(total_logs))

    # Statistiques par méthode HTTP
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    # Nombre de requêtes d'état (GET sur /status)
    status_checks = collection.count_documents(
        {"method": "GET", "path": "/status"})
    print("{} status check".format(status_checks))


if __name__ == "__main__":
    log_stats()
