
import numpy as np
import matplotlib.pyplot as plt


class PSO:
    def __init__(self, lower_bound, upper_bound, swarm_size, no_features, no_iterations, fitness_func):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.swarm_size = swarm_size
        self.no_features = no_features
        self.no_iterations = no_iterations
        self.fitness_function = fitness_func

        self.particles = None
        self.velocities = None
        self.best_pos_personal = None
        self.best_fitness_personal = None
        self.global_best_pos = None
        self.global_best_fitness = float('inf')



    def initialize_particles(self):
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.no_features))
        self.velocities = np.zeros((self.swarm_size, self.no_features))
        self.best_pos_personal = self.particles.copy()
        self.best_fitness_personal = np.zeros(self.swarm_size)
        self.global_best_pos = np.zeros(self.no_features)

    def start(self):
        self.initialize_particles()

        for _ in range(self.no_iterations):
            # Aktualizacja najlepszej globalnej pozycji
            global_best_index = np.argmin(self.best_fitness_personal)
            if self.best_fitness_personal[global_best_index] < self.global_best_fitness:
                self.global_best_pos = self.best_pos_personal[global_best_index].copy()
                self.global_best_fitness = self.best_fitness_personal[global_best_index]

            for j in range(self.no_features):  # Iteracja po wszystkich wymiarach
                for i in range(self.swarm_size):
                    # Aktualizacja pozycji
                    self.particles[i, j] = np.random.uniform(self.lower_bound, self.upper_bound)

                    # Aktualizacja najlepszych lokalnych pozycji
                    j_fitness_value = self.fitness_function(self.particles[i, j])
                    if j_fitness_value < self.best_fitness_personal[i]:
                        self.best_pos_personal[i, j] = self.particles[i, j]
                        self.best_fitness_personal[i] = j_fitness_value

                    # Aktualizacja prędkości
                    r1 = np.random.uniform(0, 1)
                    r2 = np.random.uniform(0, 1)
                    inertia_weight = 0.5
                    cognitive_factor = 0.5
                    social_factor = 0.5
                    self.velocities[i, j] = (inertia_weight * self.velocities[i, j] +
                                             cognitive_factor * r1 * (
                                                     self.best_pos_personal[i, j] - self.particles[i, j]) +
                                             social_factor * r2 * (self.global_best_pos[j] - self.particles[i, j]))

                    # Ograniczenie prędkości
                    self.velocities[i, j] = np.clip(self.velocities[i, j], self.lower_bound - self.upper_bound,
                                                    self.upper_bound - self.lower_bound)

                # Aktualizacja pozycji cząstek
                self.particles[:, j] += self.velocities[:, j]

        # Wypisanie rozwiązania
        print("Rozwiązanie g:", self.global_best_pos)


