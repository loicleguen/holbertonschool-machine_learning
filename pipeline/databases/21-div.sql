-- Script qui crée la fonction SafeDiv pour diviser deux nombres en évitant la division par zéro
DELIMITER //
CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN a / b;
    END IF;
END;
//
DELIMITER ;
