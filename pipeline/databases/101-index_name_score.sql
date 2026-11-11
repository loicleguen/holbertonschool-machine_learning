-- Script SQL qui crée un index composite idx_name_first_score sur la table names
-- indexant le premier caractère du nom (name(1)) ET le score
CREATE INDEX idx_name_first_score ON names(name(1), score);
