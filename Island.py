from GA_main import GeneticAlgorithm, Specimen
from typing import Tuple
import random
import numpy as np

class Islands:
    def __init__(self, num_islands: int, 
                 population_size: int, 
                 mutation_rate: float,
                 number_of_objects: int,
                 max_number_iterations: int,
                 stagnation: float,
                 migration_interval: int,
                 population_stagnation : int):
        """
        Создание архипелага островов с собственными популяциями (островная модель ГА)

        Args:
            num_islands: количество островов
            population_size: размер популяции на каждом острове
            mutation_rate: вероятность мутации
            number_of_objects: размерность вектора (число объектов)
            max_number_iterations: ограничение на общее число поколений
            stagnation: epsilon — критерий остановки по стагнации
            migration_interval: каждые N итераций запускается миграция
        """
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.max_number_iterations = max_number_iterations
        self.stagnation = stagnation
        self.population_stagnation = population_stagnation

        # создаем острова
        self.islands = [
            GeneticAlgorithm(
                population_size=population_size,
                mutation_rate=mutation_rate,
                number_of_objects=number_of_objects,
                max_number_iterations=max_number_iterations,
                stagnation=stagnation
            )
            for _ in range(num_islands)
        ]

        # генерируем популяции на всех островах
        for island in self.islands:
            island.generate_population()

    def migrate(self):
        """
        Миграция особей между островами по кольцу с элитизмом.
        С каждого острова:
         - лучшая особь (min fitness) - копируется
         - случайная особь (из оставшихся) - копируется
        Переходят на следующий остров (последний → первый).
        Элитные особи остаются на исходном острове.
        После добавления мигрирующих особей удаляются две худшие особи для сохранения размера популяции.
        """
        best_and_random_pairs = []

        # собираем пары для миграции
        for island in self.islands:
            best_fitness, best_specimen = island.find_best_specimen()
            # случайная особь, отличная от лучшей
            other_specimens = [s for s in island.population if s is not best_specimen]
            random_specimen = random.choice(other_specimens)
            best_and_random_pairs.append((best_specimen, random_specimen))

        # миграция по кольцу
        for i, island in enumerate(self.islands):
            next_island = self.islands[(i + 1) % self.num_islands]
            best, rand = best_and_random_pairs[i]

            # Создаем копии особей для миграции (элитизм - оригиналы остаются)
            best_copy = Specimen(best.vector.copy(), best.fitness)
            rand_copy = Specimen(rand.vector.copy(), rand.fitness)

            # Добавляем копии на следующий остров
            next_island.population.extend([best_copy, rand_copy])
            
            # Удаляем две худшие особи для сохранения размера популяции
            # Сортируем популяцию по приспособленности (по возрастанию - худшие в конце)
            next_island.population.sort(key=lambda x: x.fitness)
            # Удаляем две худшие особи
            next_island.population = next_island.population[:-2]

    def run(self):
        """
        Запускает эволюцию всех островов с учетом миграции и критериев остановки.
        Возвращает лучшую особь из всех островов.
        """
        prev_avg_fitness = float('inf')

        stagnation_counter = 0

        for iteration in range(1, self.max_number_iterations + 1):
            # выполняем одну итерацию на каждом острове
            for island in self.islands:
                island.run_one_iteration()

            # миграция по интервалу
            if iteration % self.migration_interval == 0:
                self.migrate()
                print(f"--- Миграция после {iteration} поколений ---")

            # средняя приспособленность по всем островам
            avg_fitness_all = np.mean([island.population_fitness() for island in self.islands])

            # проверяем критерий стагнации
            if abs(prev_avg_fitness - avg_fitness_all) < self.stagnation:
                stagnation_counter += 1
                if stagnation_counter >= self.population_stagnation:
                    print(f"Остановка по стагнации на итерации {iteration}")
                    break
            else:
                stagnation_counter = 0

            prev_avg_fitness = avg_fitness_all

            if iteration % 10 == 0 or iteration == 1:
                best_fitness, _ = self.find_global_best()
                print(f"[{iteration}] Средняя приспособленность = {avg_fitness_all}, лучшая = {best_fitness}")


        # возвращаем лучший результат из всех островов
        return self.find_global_best()

    def find_global_best(self) -> Tuple[float, Specimen]:
        """
        Возвращает лучшую особь среди всех островов.
        """
        best_overall_fitness = float('inf')
        best_specimen = None

        for island in self.islands:
            best_fitness, specimen = island.find_best_specimen()
            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_specimen = specimen

        return best_overall_fitness, best_specimen


def parallel_genetic_algorithm():
    num_islands = 10
    population_size = 15
    mutation_rate = 0.1
    number_of_objects = 25
    max_number_iterations = 3000
    stagnation = 0.5
    migration_interval = 10
    population_stagnation = 10

    a = Islands(num_islands, 
                    population_size, 
                    mutation_rate,
                    number_of_objects,
                    max_number_iterations,
                    stagnation,
                    migration_interval,
                    population_stagnation)

    a.run()

if __name__ == "__main__":
    parallel_genetic_algorithm()