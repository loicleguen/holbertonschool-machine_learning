-- Affiche la température moyenne (en Fahrenheit) par ville, triée par température décroissante
SELECT city, AVG(value) AS avg_temp
FROM temperatures
GROUP BY city
ORDER BY avg_temp DESC;
