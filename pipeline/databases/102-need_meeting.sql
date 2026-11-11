-- Script SQL qui crée la vue need_meeting
-- Liste les étudiants ayant un score < 80 et aucune réunion ou une réunion de plus d'1 mois
CREATE VIEW need_meeting AS
SELECT name
FROM students
WHERE score < 80
  AND (last_meeting IS NULL OR last_meeting < ADDDATE(CURDATE(), INTERVAL -1 MONTH));
