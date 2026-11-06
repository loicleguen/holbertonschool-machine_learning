-- Script qui crée un trigger réduisant la quantité d un article après une nouvelle commande
Delimiter //
CREATE trigger decreases_quantity
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE name = NEW.item_name;
END;
//
DELIMITER ;
