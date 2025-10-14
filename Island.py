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
            
        # Отслеживание стагнации для каждого острова отдельно
        self.island_stagnation_counters = [0] * num_islands
        self.island_prev_fitness = [float('inf')] * num_islands
        self.active_islands = list(range(num_islands))  # Индексы активных островов
        self.completed_results = []  # Массив для хранения результатов остановленных островов

    def migrate(self):
        """
        Миграция особей между островами по кольцу с элитизмом.
        Участвуют только активные острова (не достигшие стагнации).
        С каждого активного острова:
         - лучшая особь (min fitness) - копируется
         - случайная особь (из оставшихся) - копируется
        Переходят на следующий активный остров.
        Элитные особи остаются на исходном острове.
        После добавления мигрирующих особей удаляются две худшие особи для сохранения размера популяции.
        """
        if len(self.active_islands) < 2:
            return  # Недостаточно активных островов для миграции
            
        best_and_random_pairs = []

        # собираем пары для миграции только с активных островов
        for island_idx in self.active_islands:
            island = self.islands[island_idx]
            best_fitness, best_specimen = island.find_best_specimen()
            # случайная особь, отличная от лучшей
            other_specimens = [s for s in island.population if s is not best_specimen]
            random_specimen = random.choice(other_specimens)
            best_and_random_pairs.append((best_specimen, random_specimen))

        # миграция по кольцу между активными островами
        for i, island_idx in enumerate(self.active_islands):
            next_island_idx = self.active_islands[(i + 1) % len(self.active_islands)]
            next_island = self.islands[next_island_idx]
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
        Стагнация отслеживается для каждого острова отдельно.
        Возвращает лучшую особь из всех островов и завершенных результатов.
        """
        for iteration in range(1, self.max_number_iterations + 1):
            # выполняем одну итерацию только на активных островах
            for island_idx in self.active_islands:
                island = self.islands[island_idx]
                island.run_one_iteration()
                
                # проверяем стагнацию для каждого острова отдельно
                current_fitness = island.population_fitness()
                prev_fitness = self.island_prev_fitness[island_idx]
                
                if abs(current_fitness - prev_fitness) < self.stagnation:
                    self.island_stagnation_counters[island_idx] += 1
                    if self.island_stagnation_counters[island_idx] >= self.population_stagnation:
                        # Остров достиг стагнации - сохраняем результат и исключаем из активных
                        best_fitness, best_specimen = island.find_best_specimen()
                        self.completed_results.append({
                            'island_idx': island_idx,
                            'best_fitness': best_fitness,
                            'best_specimen': best_specimen,
                            'iteration': iteration,
                            'reason': 'stagnation'
                        })
                        self.active_islands.remove(island_idx)
                        print(f"Остров {island_idx} остановлен по стагнации на итерации {iteration}, лучшая приспособленность = {best_fitness}")
                else:
                    self.island_stagnation_counters[island_idx] = 0
                    
                self.island_prev_fitness[island_idx] = current_fitness

            # Проверяем, остались ли активные острова
            if len(self.active_islands) == 0:
                print(f"Все острова остановлены на итерации {iteration}")
                break

            # миграция по интервалу (только между активными островами)
            if iteration % self.migration_interval == 0 and len(self.active_islands) > 1:
                self.migrate()
                print(f"--- Миграция после {iteration} поколений (активных островов: {len(self.active_islands)}) ---")

            # средняя приспособленность только по активным островам
            if len(self.active_islands) > 0:
                avg_fitness_active = np.mean([self.islands[i].population_fitness() for i in self.active_islands])
                
                if iteration % 10 == 0 or iteration == 1:
                    best_fitness, _ = self.find_global_best()
                    print(f"[{iteration}] Активных островов: {len(self.active_islands)}, средняя приспособленность = {avg_fitness_active:.2f}, лучшая = {best_fitness:.2f}")

        # Сохраняем результаты всех активных островов
        for island_idx in self.active_islands:
            best_fitness, best_specimen = self.islands[island_idx].find_best_specimen()
            self.completed_results.append({
                'island_idx': island_idx,
                'best_fitness': best_fitness,
                'best_specimen': best_specimen,
                'iteration': self.max_number_iterations,
                'reason': 'max_iterations'
            })

        # возвращаем лучший результат из всех завершенных островов
        return self.find_global_best()

    def find_global_best(self) -> Tuple[float, Specimen]:
        """
        Возвращает лучшую особь среди всех островов (включая завершенные результаты).
        """
        best_overall_fitness = float('inf')
        best_specimen = None

        # Проверяем результаты завершенных островов
        for result in self.completed_results:
            if result['best_fitness'] < best_overall_fitness:
                best_overall_fitness = result['best_fitness']
                best_specimen = result['best_specimen']

        # Проверяем активные острова
        for island_idx in self.active_islands:
            island = self.islands[island_idx]
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
    stagnation = 0.1
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