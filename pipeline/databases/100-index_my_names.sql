-- Script SQL qui crée un index idx_name_first sur la table names
-- en indexant uniquement le premier caractère de la colonne name
CREATE INDEX idx_name_first ON names(name(1));
