#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Style global pour rendre les graphiques plus modernes
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ==========================================
# 1. Graphique linéaire (Line Plot)
# ==========================================
def exemple_courbe():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y1, label='Sinus', color='#1f77b4', linewidth=2)
    plt.plot(x, y2, label='Cosinus', color='#ff7f0e', linestyle='--', linewidth=2)
    
    plt.title('1. Courbes Sinus et Cosinus', fontsize=14, fontweight='bold')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# ==========================================
# 2. Diagramme en barres (Bar Chart)
# ==========================================
def exemple_barres():
    categories = ['Python', 'C++', 'Java', 'JavaScript', 'R']
    utilisateurs = [85, 45, 60, 75, 30]
    couleurs = ['#2b5c8f', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(categories, utilisateurs, color=couleurs, edgecolor='black', alpha=0.85)

    # Ajout des valeurs au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval}%', ha='center', va='bottom', fontweight='bold')

    plt.title('2. Utilisation des Langages en Data Science', fontsize=14, fontweight='bold')
    plt.xlabel('Langage')
    plt.ylabel('Utilisation (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. Nuage de points (Scatter Plot)
# ==========================================
def exemple_nuage_points():
    np.random.seed(42)
    x = np.random.rand(50) * 100
    y = x * 1.2 + np.random.normal(0, 15, 50)
    tailles = np.random.rand(50) * 300 + 50
    couleurs = np.random.rand(50)

    plt.figure(figsize=(8, 4.5))
    scatter = plt.scatter(x, y, s=tailles, c=couleurs, cmap='viridis', alpha=0.7, edgecolors='w', linewidth=1.5)
    
    plt.colorbar(scatter, label='Intensité')
    plt.title('3. Corrélation et Distribution (Scatter Plot)', fontsize=14, fontweight='bold')
    plt.xlabel('Variable X')
    plt.ylabel('Variable Y')
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Diagramme circulaire / Donut (Pie Chart)
# ==========================================
def exemple_donut():
    labels = ['Ventes Directes', 'Partenaires', 'En Ligne', 'Autres']
    tailles = [40, 25, 20, 15]
    couleurs = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    explode = (0.05, 0, 0, 0)  # Fait sortir le 1er segment

    plt.figure(figsize=(6, 6))
    plt.pie(tailles, explode=explode, labels=labels, colors=couleurs, autopct='%1.1f%%',
            startangle=140, pctdistance=0.75, textprops={'fontsize': 11})

    # Transformer le Pie Chart en Donut Chart
    centre_circle = plt.Circle((0,0), 0.55, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title('4. Répartition du Chiffre d\'Affaires', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. Multi-graphiques (Subplots 2x2)
# ==========================================
def exemple_subplots():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    
    x = np.linspace(0, 5, 50)
    
    # Subplot 1
    axes[0, 0].plot(x, x**2, color='red')
    axes[0, 0].set_title('Quadratique ($x^2$)')
    
    # Subplot 2
    axes[0, 1].scatter(x, np.sqrt(x), color='green')
    axes[0, 1].set_title('Racine Carrée ($\sqrt{x}$)')
    
    # Subplot 3
    axes[1, 0].hist(np.random.randn(500), bins=20, color='purple', alpha=0.7)
    axes[1, 0].set_title('Histogramme (Loi Normale)')
    
    # Subplot 4
    axes[1, 1].boxplot(np.random.randn(50, 4), patch_artist=True)
    axes[1, 1].set_title('Boîtes à Moustaches (Boxplot)')
    
    fig.suptitle('5. Combinaison de Plusieurs Graphiques', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Exécution des exemples Matplotlib...")
    exemple_courbe()
    exemple_barres()
    exemple_nuage_points()
    exemple_donut()
    exemple_subplots()