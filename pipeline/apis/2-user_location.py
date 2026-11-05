#!/usr/bin/env python3
"""Module pour obtenir la localisation d'un utilisateur GitHub."""
import requests
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) > 1:
        res = requests.get(sys.argv[1])
        if res.status_code == 200:
            print(res.json().get('location'))
        elif res.status_code == 404:
            print('Not found')
        elif res.status_code == 403:
            reset = int(res.headers.get('X-Ratelimit-Reset', 0))
            minutes = int((reset - time.time()) / 60)
            print(f"Reset in {minutes} min")
