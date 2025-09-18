#тестовое изменение

from typing import List, Tuple, Callable
import random
import numpy as np


class GeneticAlgorithm:

    def __init__(self) -> None:
        '''
        Создание объекта класса GeneticAlgorithm
        '''
        self.cost_matrix = random_matrix_15x15 = np.random.randint(1, 11, size=(15, 15))#[[1, 2, 3, 4, 5, 6, 7, 8],
        #                     [2, 3, 4, 5, 6, 7, 8, 1],
        #                     [3, 4, 5, 6, 7, 8, 1, 2],
        #                     [4, 5, 6, 7, 8, 1, 2, 3],
        #                     [5, 6, 7, 8, 1, 2, 3, 4],
        #                     [6, 7, 8, 1, 2, 3, 4, 5],
        #                     [7, 8, 1, 2, 3, 4, 5, 6],
        #                     [8, 1, 2, 3, 4, 5, 6, 7]]
        # self.n = len(self.cost_matrix[0]) ?? всегда так ? мне так кажется друзья!
        self.population_size = 50
        self.population = []
        self.min_fitness = float('inf')

    #def print_all_specimen(self):


    def generation_individual(self) -> List[int]: #тут разве не n = len(self.cost_matrix[0])  ???
        """
        Генерирует случайную особь - допустимое решение задачи о назначениях.

        Алгоритм:
        1. Создается перестановка чисел от 1 до n с помощью np.random.permutation
        2. К каждому элементу добавляется 1 (так как permutation возвращает значения от 0 до n-1)
        3. Возвращается вектор назначений = позиция - объект, число - номер назначенного ресурса
        """

        #if n is None:
        n = len(self.cost_matrix[0])
        x = list(range(1,n+1))
        random.shuffle(x)
        # x = np.random.permutation(n) + 1
        return x
        # return x.tolist()

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

    def is_acceptable(self, solution: List[int]) -> bool:
        """
        Проверка решения на допустимость

        Функция реализует проверку уникальности элементов.
        Использует сравнение длины списка и множества для проверки уникальности.
        (В множестве хранятся только уникальные элементы).

        Parameters:
        solution : List[int] - особь (массив чисел для проверки)

        Returns: bool
            True - решение допустимо,
            False - решение недопустимо
        """
        return len(solution) == len(set(solution))

    def fitness_func(self, specimen: list) -> int:
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
        for k in range(len(self.cost_matrix[0])):
            result_fitness += self.cost_matrix[specimen[k] - 1][k]  

        return result_fitness

    def fitness_population_func(self) -> int:
        '''
        Подсчет приспособленности популяции
        Returns: 
        Алгоритм: вычисляем приспособленность для каждой особи, после чего выбираем минимальное значение.
        '''
        if not self.population:
            return float('inf')
    
        min_fitness = float('inf')
        for individual in self.population:
            current_fitness = self.fitness_func(individual)
            # if current_fitness < min_fitness:
            min_fitness = min(current_fitness,min_fitness)
        return min_fitness

    def mutation_assignment(self, solution: List[int], mutation_rate: float = 0.1) -> List[int]:
        """
        Выполняет мутацию для особи.

        Мутация заключается в случайной перестановке двух элементов в решении,
        что сохраняет допустимость решения (все элементы остаются уникальными).

        Parameters:
        solution : List[int] - Текущее решение
        mutation_rate : float, optional - Вероятность применения мутации к решению (по умолчанию 0.1)

        Returns:
        List[int] - Мутированное решение (может быть идентично исходному если мутация не применена)
        """

        # Создаем копию решения чтобы не изменять оригинал
        mutated_solution = solution.copy()

        # Проверяем, нужно ли применять мутацию
        if random.random() < mutation_rate:
            # Выбираем два случайных различных индекса
            idx1, idx2 = random.sample(range(len(solution)), 2)

            # Меняем местами элементы на выбранных позициях
            mutated_solution[idx1], mutated_solution[idx2] = mutated_solution[idx2], mutated_solution[idx1]

        return mutated_solution


    def selection_mu_plus_lambda(self, parents: List[List[int]], offspring: List[List[int]],
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

        size = len(parent1)

        # Выбираем одну случайную точку разрыва
        crossover_point = random.randint(1, size - 1)

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

        size = len(main_parent)
        child = [-1] * size

        # Копируем сегмент от начала до точки разрыва из основного родителя
        child[:crossover_point] = main_parent[:crossover_point]

        # Создаем множество уже использованных генов для быстрой проверки
        used_genes = set(child[:crossover_point])

        # Заполняем оставшиеся позиции, проходя по всему второму родителю
        current_idx = crossover_point

        for gene in secondary_parent:
            # Если уже заполнили все позиции - выходим
            if current_idx >= size:
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
        parents = random.sample(self.population, 2)
        
        # 3. Скрещиваем
        offspring1, offspring2 = self.crossover(parents[0], parents[1])
        print(len(parents))
#        print(len(offspring2))
        # # 4. Мутируем потомков
        # for i in self.population:
        #     self.mutation_assignment(i)
        offspring1 = self.mutation_assignment(offspring1)
        offspring2 = self.mutation_assignment(offspring2)
        
        # 5. Отбор (µ + λ)
        new_population = self.selection_mu_plus_lambda(
            self.population, 
            [offspring1, offspring2],
            self.fitness_func,
            self.population_size
        )
        
        # 6. Обновляем популяцию
        self.population = new_population
        
        # 7. Возвращаем результаты
        best_fitness = self.fitness_population_func()
        return best_fitness, self.population

def main():
    print("HI!")
     # Создаем объект алгоритма
    ga = GeneticAlgorithm()
    
    # Генерируем начальную популяцию
    ga.generate_population()
    print(ga.fitness_population_func())
    for _ in range (5000):
        best, arr = ga.run_one_iteration()

    print(best)

if __name__ == "__main__":
    main()