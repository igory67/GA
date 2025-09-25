from typing import List, Tuple
import random
import numpy as np

"""
    Знать основные структуры хранения данных и их методы, используемые в коде.
                
    Шаблон описания функции:
        Название:
        Параметры:
        Алгоритм:
        Что возвращает:

    Осталось:
1. Функция отбора в новую популяцию (добавь в run_one_iteration(self))
2. Приспособленность популяция (изменить с минимального на среднее) - ???
3. Нужно ли хранить самую приспособленную особь?
4. Критерии остановки (есть только T, эпсилон нет)
Все остальное вроде не срочно
"""

class Specimen:
    def __init__(self, vector: List[int], fitness: float = float('inf')):
        self.vector = vector # допустимое решение
        self.fitness = fitness # приспособленность особи


class GeneticAlgorithm:

    def __init__(self, population_size, mutation_rate, max_number_iterations, stagnation) -> None:
        '''
        Создание объекта класса GeneticAlgorithm
        '''
        self.mutation_rate = mutation_rate # p_m - вероятность мутации
        self.population_size = population_size # N - размер популяции
        self.max_number_iterations = max_number_iterations # T - максимальное количество итераций
        self.stagnation = stagnation # epsilon - разницы между поколениями
        self.vector_size = 20 # n - количество предприятий (ресурсов)

        # данные для заполнения матрицы стоимостей случайными значениями
        self.min_cost = 1 # минимальное значение матрицы
        self.max_cost = 100 # максимальное значение матрицы

        self.cost_matrix = np.random.randint(self.min_cost, self.max_cost, 
                                             size=(self.vector_size, self.vector_size)) # заполнение матрицы
        self.population: List[Specimen] = [] # популяция (массив особей)

    def generation_individual(self) -> Specimen:
        """
        Генерирует случайную особь - допустимое решение задачи о назначениях.

        Алгоритм:
        1. Создается перестановка чисел от 0 до n-1 с помощью np.random.permutation
        2. Возвращается объект Specimen с полученной перестановкой (преобразованной в список) в виде значений.
        """
        x = np.random.permutation(self.vector_size)
        return Specimen(x.tolist())

    def generate_population(self) -> List[Specimen]:
        """
        Генерирует популяцию особей 

        Алгоритм:
        1. Создается особь (self.population_size=N) раз
        2. Каждой особи рассчитывается приспособленность
        3. Возвращается популяция особей.
        """
        self.population = []
        for _ in range (self.population_size): #генерируем N раз
            # генерируем особь
            specimen = self.generation_individual()

            # считаем приспособленность особи и записываем значение в соответствующее поле класса особи
            self.specimen_fitness(specimen)

            # добавляем особь в популяцию (в список)
            self.population.append(specimen)

        return self.population

    def is_acceptable(self, specimen: Specimen) -> bool:
        """
        Проверка решения на допустимость

        Функция реализует проверку уникальности элементов вектора особи.
        Использует сравнение длины списка и множества (полученного из этого списка) для проверки уникальности его элементов.
        (В множестве хранятся только уникальные элементы, поэтому длины будут равны только если в списке тоже только уникальные аргументы).

        Params:
            Specimen: - особь для проверки
            self: - сам класс
        
        Return: 
            bool:
            True - решение допустимо,
            False - решение недопустимо
        """
        return (len(specimen.vector) == len(set(specimen.vector)))

    def specimen_fitness(self, specimen: Specimen) -> int:
        '''
        Подсчет приспособленности особи

        Параметры:
        specimen: Особь для подсчета приспособленности
        Значение функции приспособленности особи

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
            result_fitness += self.cost_matrix[specimen.vector[i]][i]
        
        specimen.fitness = result_fitness
        return result_fitness
    
    def population_fitness(self) -> float:
        '''
        Подсчет средней приспособленности популяции.
        Returns:
        float: Среднее значение приспособленности по всей популяции.
        '''
        if not self.population:
            return 0.0
        
        total_fitness = sum(specimen.fitness for specimen in self.population)
        return total_fitness / len(self.population)
    
    def find_best_specimen(self) -> Tuple[int, Specimen]:
        '''
        Подсчет приспособленности популяции
        Returns:
        min_fitness - значение лучшей приспособленности
        best_specimen - самая приспособленная особь
        Алгоритм: выбираем лучшую особь из популяции по ее приспособленности.
        '''
        min_fitness = float('inf')

        for specimen in self.population:
            if specimen.fitness < min_fitness:
                min_fitness = specimen.fitness
                best_specimen = specimen

        return min_fitness, best_specimen
    
    def mutation_specimen(self, specimen: Specimen) -> Specimen:
        """
        Выполняет мутацию для особи.

        Мутация заключается в случайной перестановке двух элементов в решении,
        что сохраняет допустимость решения (все элементы остаются уникальными).

        Parameters:
        specimen : Specimen - Текущая особь

        Returns:
        Specimen - Мутированная особь (может быть идентична исходной если мутация не применена)
        """

        mutated_vector = specimen.vector.copy()

        # Проверяем, нужно ли применять мутацию
        if random.random() < self.mutation_rate:
            # Выбираем два случайных различных индекса
            i, j = random.sample(range(self.vector_size), 2)

            # Меняем местами элементы на выбранных позициях
            mutated_vector[i], mutated_vector[j] = mutated_vector[j], mutated_vector[i]
        
        mutated_specimen = Specimen(mutated_vector)
        # считаем приспособленность особи и записываем значение в поле
        self.specimen_fitness(mutated_specimen)
        return mutated_specimen

    def shaping_next_generation(self, parents: List[Specimen], offspring: List[Specimen]) -> List[Specimen]:
        """
        Выполняет отбор особей для формирования следующего поколения.

        Из объединенного пула родителей и потомков выбираются лучшие особи
        на основе значения fitness.

        Parameters:
        parents : List[Specimen] - Список родителей текущего поколения
        offspring : List[Specimen] - Список потомков

        Returns:
        List[Specimen] - Список лучших особей для следующего поколения
        """
        # Объединяем родителей и потомков в один пул
        combined_pool = parents + offspring

        # Сортируем объединенный пул по значению fitness-функции (чем меньше, тем лучше)
        sorted_pool = sorted(combined_pool, key=lambda s: s.fitness)

        # Выбираем mu лучших особей из отсортированного пула
        selected_individuals = sorted_pool[:self.population_size]

        return selected_individuals

    def crossover(self, parent1: Specimen, parent2: Specimen) -> Specimen:
        """
        Кроссовер. Создание потомка из двух родителей с одной точкой разрыва.

        Кроссовер сохраняет сегмент от начала до точки разрыва из первого родителя
        и заполняет оставшиеся позиции генами из второго родителя в порядке их следования,
        пропуская уже присутствующие гены.

        Parameters:
        parent1 : Specimen - Первый родитель
        parent2 : Specimen - Второй родитель

        Returns:
        Specimen - потомок
        """
        # Выбираем одну случайную точку разрыва
        crossover_point = random.randint(0, self.vector_size - 1)

        # Создаем вектор потомка
        child_vector = self.create_child_vector(parent1.vector, parent2.vector, crossover_point)

        child = Specimen(child_vector)

        # считаем приспособленность особи и записываем значение в поле
        self.specimen_fitness(child)

        return child

    def create_child_vector(self, main_parent_vector: List[int], secondary_parent_vector: List[int],
                            crossover_point: int) -> List[int]:
        """
        Создание вектора одного потомка из векторов двух родителей с заданной точкой разрыва.

        Parameters:
        main_parent_vector : List[int] - Вектор родителя, от которого берется начальный сегмент
        secondary_parent_vector : List[int] - Вектор родителя, от которого берутся оставшиеся гены
        crossover_point : int - Точка разрыва

        Returns:
        List[int] - Вектор потомка
        """
        child_vector = [-1] * self.vector_size

        # Копируем сегмент от начала до точки разрыва из основного родителя
        child_vector[:crossover_point] = main_parent_vector[:crossover_point]
        
        # Создаем множество уже использованных генов для быстрой проверки
        used_genes = set(child_vector[:crossover_point])

        # Заполняем оставшиеся позиции, проходя по всему второму родителю
        current_idx = crossover_point

        for gene in secondary_parent_vector:
            # Если уже заполнили все позиции - выходим
            if current_idx >= self.vector_size:
                break

            # Если гена еще нет в потомке - добавляем
            if gene not in used_genes:
                child_vector[current_idx] = gene
                used_genes.add(gene)
                current_idx += 1

        return child_vector

    def hamming_distance(self, vector1: List[int], vector2: List[int]) -> int:
        """
        Вычисляет расстояние Хэмминга между двумя векторами.
        Расстояние Хэмминга - это количество позиций, в которых соответствующие символы различны.
        """
        return sum(el1 != el2 for el1, el2 in zip(vector1, vector2))

    def select_parents_inbreeding(self) -> Tuple[Specimen, Specimen]:
        """
        Выбирает пару родителей с использованием инбридинга.
        1. Первый родитель выбирается случайным образом.
        2. Для всех остальных особей вычисляется расстояние Хэмминга до первого родителя.
        3. Вторым родителем становится особь с минимальным расстоянием Хэмминга.
        """
        parent1 = random.choice(self.population)
        
        other_specimens = [s for s in self.population if s is not parent1]
        
        if not other_specimens:
            # Если в популяции только одна особь, возвращаем ее дважды
            return parent1, parent1

        min_dist = float('inf')
        parent2 = None

        for specimen in other_specimens:
            dist = self.hamming_distance(parent1.vector, specimen.vector)
            if dist < min_dist:
                min_dist = dist
                parent2 = specimen
        
        return parent1, parent2

    def roulette_selection(self):
        """
        Улучшенная версия с явным построением рулетки
        """
        
        # 1. Используем ранжирование вместо инвертирования
        fitness_values = [specimen.fitness for specimen in self.population]

        # Преобразуем для минимизации (меньше fitness = лучше)
        max_fitness = max(fitness_values)
        transformed_fitness = [max_fitness - f + 1e-6 for f in fitness_values]

        total_fitness = sum(transformed_fitness)
        expected_counts = [f * len(self.population) / total_fitness for f in transformed_fitness]

        # 2. Remainder Stochastic Sampling
        pre_new_population = []

        # Гарантированная часть + дробная
        for i, count in enumerate(expected_counts):
            integer_part = int(count)
            for _ in range(integer_part):
                pre_new_population.append((self.population[i], i))
            
            frac_part = count - int(count)
            pointer = random.uniform(0, 1)
            if pointer <= frac_part:
                pre_new_population.append((self.population[i], i))
        
        new_population = []
        for _ in range (len(self.population)):
            pointer = random.randint(0, len(pre_new_population) - 1)
            new_gen = self.population[pre_new_population[pointer][1]]
            new_population.append(new_gen)
        
        self.population = new_population

    def run_one_iteration(self) -> Tuple[int, Specimen]:
        """
        Запускает одну итерацию генетического алгоритма.
        
        Returns:
        Tuple[int, Specimen] - лучшая приспособленность и лучшая особь
        """
        # 1. Проверяем, сгенерирована ли популяция
        if not self.population:
            self.generate_population()
        

        offspring_population = [] # массив потомков
        num_pairs = self.population_size // 2 # количество пар

        # 2. Скрещиваем n // 2 пар 
        for _ in range(num_pairs):
            # выбираем два родителя
            parent1, parent2 = self.select_parents_inbreeding()
            
            # скрещиваем родителей, получаем двух потомков
            offspring = self.crossover(parent1, parent2)

            offspring = self.mutation_specimen(offspring)
            
            offspring_population.append(offspring)

        # 3. Отбор в новую популяцию
        self.roulette_selection()
        
        # 4. Возвращаем результаты
        best_fitness, best_specimen = self.find_best_specimen()

        return best_fitness, best_specimen


def main():
    print("HI!")
     # Создаем объект алгоритма
    ga = GeneticAlgorithm(population_size = 30, mutation_rate = 0.2, max_number_iterations = 500, stagnation = 0)
    
    # Генерируем начальную популяцию
    ga.generate_population()
    initial_fitness, initial_specimen = ga.find_best_specimen()
    print(f"Начальная приспособленность: {initial_fitness}, лучшая особь: {initial_specimen.vector}")

    for i in range (1, ga.max_number_iterations):
        best_fitness, best_specimen = ga.run_one_iteration()
        if i % 10 == 0:
            print(f"Приспособленность на {i} шаге: {best_fitness}, лучшая особь: {best_specimen.vector}")

if __name__ == "__main__":
    main()