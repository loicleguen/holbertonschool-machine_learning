-- Script qui crée la procédure stockée AddBonus
DELIMITER //
CREATE PROCEDURE AddBonus(
    IN user_id INT,
    IN project_name VARCHAR(255),
    IN score INT
)
BEGIN
    DECLARE p_id INT;

    -- Vérifier si le projet existe déjà
    SELECT id INTO p_id
    FROM projects
    WHERE name = project_name;

    -- Si le projet n'existe pas (p_id est NULL), on le crée
    IF p_id IS NULL THEN
        INSERT INTO projects (name) VALUES (project_name);
        SET p_id = LAST_INSERT_ID();
    END IF;

    -- Ajouter la nouvelle correction
    INSERT INTO corrections (user_id, project_id, score)
    VALUES (user_id, p_id, score);
END;
//
DELIMITER ;
