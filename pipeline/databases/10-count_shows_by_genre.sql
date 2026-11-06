-- Script qui répertorie tous les genres et affiche le nombre d'émissions liées à chacun.
SELECT tv_genres.name AS genre, COUNT(tv_shows.id) as number_of_shows
FROM tv_genres
JOIN tv_show_genres ON tv_show_genres.genre_id = tv_genres.id
JOIN tv_shows ON tv_shows.id = tv_show_genres.show_id
GROUP BY genre
ORDER BY number_of_shows DESC;
