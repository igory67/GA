from typing import List, Tuple, Callable
import random
import numpy as np


class GeneticAlgorithm:

    def __init__(self, population_size, mutation_rate, max_number_iterations) -> None:
        '''
        Создание объекта класса GeneticAlgorithm
        '''
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_number_iterations = max_number_iterations
        self.vector_size = 20
        self.min_cost = 1
        self.max_cost = 100
        self.cost_matrix = np.random.randint(self.min_cost, self.max_cost, size=(self.vector_size, self.vector_size))
        self.population = []

    def generation_individual(self) -> List[int]:
        """
        Генерирует случайную особь - допустимое решение задачи о назначениях.

        Алгоритм:
        1. Создается перестановка чисел от 1 до n с помощью np.random.permutation
        2. К каждому элементу добавляется 1 (так как permutation возвращает значения от 0 до n-1)
        3. Возвращается вектор назначений = позиция - объект, число - номер назначенного ресурса
        """
        x = np.random.permutation(self.vector_size) + 1
        return x.tolist()

    def generate_population(self) -> List[List[int]]:
        """
        Генерирует популяцию особей 

        Алгоритм:
        1. Создается особь N(self.population_size) раз
        2. Возвращается вектор назначений = позиция - объект, число - номер назначенного ресурса
        """
        self.population = [] # если это первый раз, надо обдумать, надо ли вот так чистить
        for _ in range (self.population_size):
            self.population.append(self.generation_individual())
        return self.population

    def is_acceptable(self, specimen: List[int]) -> bool:
        """
        Проверка решения на допустимость

        Функция реализует проверку уникальности элементов.
        Использует сравнение длины списка и множества для проверки уникальности.
        (В множестве хранятся только уникальные элементы).

        Parameters:
        specimen : List[int] - особь (массив чисел для проверки)

        Returns: bool
            True - решение допустимо,
            False - решение недопустимо
        """
        return len(specimen) == len(set(specimen))

    def specimen_fitness(self, specimen: list) -> int:
        '''
        Подсчет приспособленности особи
        :return: Значение функции приспособленности особи

        Алгоритм: "Имеется особь X. Также известна матрица стоимости C.
        n - количество видов ресурсов/объектов
        Cij - Стоимость затрат на назначение i вида ресурса на j объект.
        Необходимо пройтись с помощью итератора k по всем значениям вектора X (от 1 до n)
        Изначально значение функции приспособленности особи равно 0
        1. на k итерации смотрим значение xk
        2. Суммируем текущее значение функции приспособленности с Cij где i = xk, j = k 
        Сложность алгоритма O(n)"
        '''
        result_fitness = 0
        # Считаем приспособленность особи
        for i in range(self.vector_size):
            result_fitness += self.cost_matrix[specimen[i] - 1][i]  

        return result_fitness

    def population_fitness(self) -> Tuple[int, List[int]]:
        '''
        Подсчет приспособленности популяции
        Returns:
        min_fitness - значение лучшей приспособленности
        best_specimen - самая приспособленная особь
        Алгоритм: вычисляем приспособленность для каждой особи, после чего выбираем минимальное значение.
        '''
        min_fitness = float('inf')

        for specimen in self.population:
            current_fitness = self.specimen_fitness(specimen)
            if current_fitness < min_fitness:
                min_fitness = current_fitness
                best_specimen = specimen

        return min_fitness, best_specimen
    
    def mutation_specimen(self, specimen: List[int], mutation_rate: float) -> List[int]:
        """
        Выполняет мутацию для особи.

        Мутация заключается в случайной перестановке двух элементов в решении,
        что сохраняет допустимость решения (все элементы остаются уникальными).

        Parameters:
        specimen : List[int] - Текущее решение
        mutation_rate : float, optional - Вероятность применения мутации к решению (по умолчанию 0.1)

        Returns:
        List[int] - Мутированное решение (может быть идентично исходному если мутация не применена)
        """

        # Создаем копию решения чтобы не изменять оригинал
        mutated_specimen = specimen.copy()

        # Проверяем, нужно ли применять мутацию
        if random.random() < mutation_rate:
            # Выбираем два случайных различных индекса
            i, j = random.sample(range(self.vector_size), 2)

            # Меняем местами элементы на выбранных позициях
            mutated_specimen[i], mutated_specimen[j] = mutated_specimen[j], mutated_specimen[i]

        return mutated_specimen

    def shaping_next_generation(self, parents: List[List[int]], offspring: List[List[int]],
                                 fitness_func: callable, mu: int) -> List[List[int]]:
        """
        Выполняет отбор особей для формирования следующего поколения.

        Из объединенного пула родителей и потомков размера (µ + λ) выбираются µ лучших особей
        на основе значения fitness-функции.

        Parameters:
        parents : List[List[int]] - Список родителей текущего поколения (размер µ)
        offspring : List[List[int]] - Список потомков (размер λ)
        fitness_func : callable - Функция оценки приспособленности (чем меньше, тем лучше)
        mu : int - Количество особей для отбора в следующее поколение

        Returns:
        List[List[int]] - Список из µ лучших особей для следующего поколения
        """

        # Объединяем родителей и потомков в один пул
        combined_pool = parents + offspring

        # Сортируем объединенный пул по значению fitness-функции (чем меньше, тем лучше)
        sorted_pool = sorted(combined_pool, key=fitness_func)

        # Выбираем mu лучших особей из отсортированного пула
        selected_individuals = sorted_pool[:mu]

        return selected_individuals

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Кроссовер. Создание двух потомков из двух родителей с одной точкой разрыва.

        Кроссовер сохраняет сегмент от начала до точки разрыва из первого родителя
        и заполняет оставшиеся позиции генами из второго родителя в порядке их следования,
        пропуская уже присутствующие гены.

        Parameters:
        parent1 : List[int] - Первый родитель
        parent2 : List[int] - Второй родитель

        Returns:
        Tuple[List[int], List[int]] - Два потомка
        """

        # Выбираем одну случайную точку разрыва
        crossover_point = random.randint(1, self.vector_size - 1) # ДВ сказала (0, size)

        # Создаем потомков
        child1 = self.create_child(parent1, parent2, crossover_point)
        child2 = self.create_child(parent2, parent1, crossover_point)

        return child1, child2

    def create_child(self, main_parent: List[int], secondary_parent: List[int],
                     crossover_point: int) -> List[int]:
        """
        Создание одного потомка из двух родителей с заданной точкой разрыва.

        Parameters:
        main_parent : List[int] - Родитель, от которого берется начальный сегмент
        secondary_parent : List[int] - Родитель, от которого берутся оставшиеся гены
        crossover_point : int - Точка разрыва

        Returns:
        List[int] - Потомок
        """
        child = [-1] * self.vector_size

        # Копируем сегмент от начала до точки разрыва из основного родителя
        child[:crossover_point] = main_parent[:crossover_point]

        # Создаем множество уже использованных генов для быстрой проверки
        used_genes = set(child[:crossover_point])

        # Заполняем оставшиеся позиции, проходя по всему второму родителю
        current_idx = crossover_point

        for gene in secondary_parent:
            # Если уже заполнили все позиции - выходим
            if current_idx >= self.vector_size:
                break

            # Если гена еще нет в потомке - добавляем
            if gene not in used_genes:
                child[current_idx] = gene
                used_genes.add(gene)
                current_idx += 1

        return child

    def run_one_iteration(self) -> Tuple[int, List[List[int]]]:
        """
        Запускает одну итерацию генетического алгоритма.
        
        Returns:
        Tuple[int, List[List[int]]] - лучшая приспособленность и новая популяция
        """
        # 1. Проверяем, сгенерирована ли популяция
        if not self.population:
            self.generate_population()
        
        # 2. Выбираем родителей (пока случайно, потом добавишь инбридинг)
        parents1, parents2 = random.sample(self.population, 2)
        
        # 3. Скрещиваем
        offspring1, offspring2 = self.crossover(parents1, parents2)

        offspring1 = self.mutation_specimen(offspring1, self.mutation_rate)
        offspring2 = self.mutation_specimen(offspring2, self.mutation_rate)
        
        # 5. Отбор (µ + λ)
        new_population = self.shaping_next_generation(
            self.population, 
            [offspring1, offspring2],
            self.specimen_fitness,
            self.population_size
        )
        
        # 6. Обновляем популяцию
        self.population = new_population
        
        # 7. Возвращаем результаты
        best_fitness, best_specimen = self.population_fitness()
        return best_fitness, best_specimen

def main():
    print("HI!")
     # Создаем объект алгоритма
    ga = GeneticAlgorithm(population_size = 30, mutation_rate = 0.2, max_number_iterations = 4000)
    
    # Генерируем начальную популяцию
    ga.generate_population()
    print(f"Начальная приспособленность: {ga.population_fitness()[0]}, лучшая особь: {ga.population_fitness()[1]}")

    for i in range (1, ga.max_number_iterations):
        best_fitness, best_specimen = ga.run_one_iteration()
        if i % 100 == 0:
            print(f"Приспособленность на {i} шаге: {best_fitness}, лучшая особь: {best_specimen}")

if __name__ == "__main__":
    main()