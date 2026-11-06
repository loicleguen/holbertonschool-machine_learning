-- Script qui classe les pays d'origine des groupes, classés par nombre de fans.
SELECT origin, SUM(fans) AS nb_fans
FROM metal_bands
GROUP BY origin
ORDER BY nb_fans DESC;
