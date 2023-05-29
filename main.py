# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from PSO import PSO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_data(dict, titleForX, titleForY, titleForPlot):
    x = list(dict.keys())
    y = list(dict.values())
    plt.plot(x, y)
    plt.xlabel(titleForX)
    plt.ylabel(titleForY)
    plt.title(titleForPlot)
    plt.show()




def sphere_function(x):
    return np.sum(x ** 2)





if __name__ == '__main__':

    max_iterations = [10, 15, 20, 25, 30, 35, 40, 50, 100, 200]
    swarm_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]
    features = [1, 2, 3, 4, 5, 6, 7]
    fitness_func_iterations = {}
    fitness_func_swarm_size = {}
    fitness_func_no_features = {}
    fitness_func_hyperparams_comb = []


#badanie wpływu liczby iteracji na wynik funkcji przystosowania
    for it in max_iterations:
            pso = PSO(lower_bound=-100, upper_bound=100, swarm_size=50, no_features=10, no_iterations=it,
                      fitness_func=sphere_function)
            pso.start()
            fitness_func_iterations.update({it : pso.fitness_function(pso.global_best_pos)})

    print(fitness_func_iterations)
    plot_data(fitness_func_iterations, "Number of iterations", "Value of Fitness Function", "Effect of the number of iterations on the fitness function value")

#badanie wpływu liczby członków populacji na wynik funkcji przystosowania
    for swarm_size in swarm_sizes:
            pso = PSO(lower_bound=-100, upper_bound=100, swarm_size=swarm_size, no_features=10, no_iterations=20,
                      fitness_func=sphere_function)
            pso.start()
            fitness_func_swarm_size.update({swarm_size :  pso.fitness_function(pso.global_best_pos)})

    print(fitness_func_swarm_size)
    plot_data(fitness_func_swarm_size, "Size of population", "Value of Fitness Function", "Effect of the size of the population on the fitness function value")

#badanie wpływu liczby cech, wymiarowości naszej funkcji na wynik funkcji przystosowania
    for no_features in features:
            pso = PSO(lower_bound=-100, upper_bound=100, swarm_size=50, no_features=no_features, no_iterations=20,
                      fitness_func=sphere_function)
            pso.start()
            fitness_func_no_features.update({no_features: pso.fitness_function(pso.global_best_pos)})

    print(fitness_func_no_features)
    plot_data(fitness_func_no_features, "Number of features/dimensions", "Value of Fitness Function",
              "Effect of the number of features/dimensions on the fitness function value")

"""

Współczynnik bezwładności (inertia_weight):

Niska wartość (bliska 0): Skutkuje większą eksploracją przestrzeni poszukiwań. 
Cząstki będą miały większą tendencję do poruszania się w różnych kierunkach, co może pomóc w poszukiwaniu nowych rozwiązań.

Wysoka wartość (bliska 1): Sprzyja eksploatacji znalezionych rozwiązań. 
Cząstki będą miały większą tendencję do poruszania się w kierunku najlepszego lokalnego lub globalnego rozwiązania,
co może przyspieszyć zbieżność.


Współczynnik poznawczy (cognitive_factor):

Niska wartość: Cząstki będą bardziej polegać na informacji lokalnej
, czyli na najlepszych dotychczasowych pozycjach, które znalazły. To może prowadzić do bardziej intensywnej eksploatacji
wokół najlepszych lokalnych rozwiązań.

Wysoka wartość: Cząstki będą bardziej polegać na informacji globalnej,
 czyli na najlepszym globalnym rozwiązaniu znalezionym w populacji.
  To może prowadzić do bardziej ekspansywnej eksploracji przestrzeni
  poszukiwań w celu znalezienia nowych i lepszych rozwiązań.
  
  
"""

#badanie wpływu róznych wartośći
inertia_weights = [0.3, 0.5, 0.9]
cognitive_factor = [0.6, 0.9, 0.2]
social_factor = [0.5, 0.7, 0.8]

#Kombinacja 1: inertia_weight = 0.3, cognitive_factor = 0.6, social_factor = 0.5
#Kombinacja 2: inertia_weight = 0.5, cognitive_factor = 0.9, social_factor = 0.7
#Kombinacja 3: inertia_weight = 0.9, cognitive_factor = 0.2, social_factor = 0.8

for i in range(len(inertia_weights)):
    pso = PSO(lower_bound=-100, upper_bound=100, swarm_size=100, no_features=4, no_iterations=20,
              fitness_func=sphere_function, inertia_weight=inertia_weights[i], cognitive_factor=cognitive_factor[i],
              social_factor=social_factor[i])

    pso.start()
    fitness_func_hyperparams_comb.append(pso.fitness_function(pso.global_best_pos))
    # Wygenerowanie wykresu
x = np.arange(len(inertia_weights))
width = 0.2

fig, ax = plt.subplots()
rects = ax.bar(x, fitness_func_hyperparams_comb, width)

# Dodanie etykiet osi i tytułu
ax.set_ylabel('Fitness function values')
ax.set_title('Effect of different PSO parameters on fitness function values')

# Dodanie etykiet na osi x
x_labels = [
    'Combination 1.',
    'Combination 2.',
    'Combination 3.'
]
ax.set_xticks(x)
ax.set_xticklabels(x_labels)

# Dodanie wartości na słupkach
for rect in rects:
    height = rect.get_height()
    ax.annotate('{}'.format(round(height, 2)), xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Utworzenie legendy
legend_patches = []
legend_labels = [
    'Inertia weight: {}, Cognitive factor: {}, Social factor: {}'.format(inertia_weights[0], cognitive_factor[0], social_factor[0]),
    'Inertia weight: {}, Cognitive factor: {}, Social factor: {}'.format(inertia_weights[1], cognitive_factor[1], social_factor[1]),
    'Inertia weight: {}, Cognitive factor: {}, Social factor: {}'.format(inertia_weights[2], cognitive_factor[2], social_factor[2])
]
for i, rect in enumerate(rects):
    color = rect.get_facecolor()
    patch = mpatches.Patch(facecolor=color, edgecolor='black', label=legend_labels[i])
    legend_patches.append(patch)

ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

# Wyświetlenie wykresuu
plt.tight_layout()
plt.show()