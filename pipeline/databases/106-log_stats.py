#!/usr/bin/env python3
"""
Module 106-log_stats
Fournit des statistiques avancees sur les logs Nginx stockes dans MongoDB,
incluant le Top 10 des IP les plus frequentes.
"""
from pymongo import MongoClient


def log_stats():
    """
    Affiche les statistiques de la collection logs.nginx:
    - Nombre total de logs
    - Nombre de logs par méthode (GET, POST, PUT, PATCH, DELETE)
    - Nombre de logs avec method=GET et path=/status
    - Top 10 des IP les plus représentées
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
        {"method": "GET", "path": "/status"}
    )
    print("{} status check".format(status_checks))

    # Top 10 des adresses IP
    print("IPs:")
    pipeline = [
        {"$group": {"_id": "$ip", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    top_ips = collection.aggregate(pipeline)
    for ip in top_ips:
        print("\t{}: {}".format(ip.get("_id"), ip.get("count")))


if __name__ == "__main__":
    log_stats()
