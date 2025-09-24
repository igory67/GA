from typing import List, Tuple, Callable
import random
import numpy as np

class Specimen:
    def __init__(self, vector: List[int], fitness: float = float('inf')):
        self.vector = vector
        self.fitness = fitness

"""
        git commit -m "убрал +1 в генерации; добавил стагнацию, узнать, как ее делать ваще выбирать, описания меняю + шаблон вот ввел"
        todo узнать всем чо такое список массив и тд тп
                
                Шаблон описания функции:
                    Название
                    Параметры:
                    Алгоритм:
                    Что возвращает

        а
"""
class GeneticAlgorithm:

    def __init__(self, population_size, mutation_rate, max_number_iterations, stagnation) -> None:
        '''
        Создание объекта класса GeneticAlgorithm
        '''
        self.mutation_rate = mutation_rate #p_m = вер. мутации
        self.population_size = population_size #N
        self.max_number_iterations = max_number_iterations #T
        self.stagnation = stagnation #epsilon разницы между поколениями
        self.vector_size = 20 #n
        self.min_cost = 1 # данные для заполнения (границы значений)
        self.max_cost = 100 # матрицы стоимостей случайными значениями
        self.cost_matrix = np.random.randint(self.min_cost, self.max_cost, size=(self.vector_size, self.vector_size)) #непосредственно заполнение
        self.population: List[Specimen] = []

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
            result_fitness += self.cost_matrix[specimen.vector[i] - 1][i]
        
        specimen.fitness = result_fitness
        return result_fitness

    def population_fitness(self) -> Tuple[int, Specimen]:
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

    def crossover(self, parent1: Specimen, parent2: Specimen) -> Tuple[Specimen, Specimen]:
        """
        Кроссовер. Создание двух потомков из двух родителей с одной точкой разрыва.

        Кроссовер сохраняет сегмент от начала до точки разрыва из первого родителя
        и заполняет оставшиеся позиции генами из второго родителя в порядке их следования,
        пропуская уже присутствующие гены.

        Parameters:
        parent1 : Specimen - Первый родитель
        parent2 : Specimen - Второй родитель

        Returns:
        Tuple[Specimen, Specimen] - Два потомка
        """
        # Выбираем одну случайную точку разрыва
        crossover_point = random.randint(1, self.vector_size - 1) # ДВ сказала (0, size)

        # Создаем векторы потомков
        child1_vector = self.create_child_vector(parent1.vector, parent2.vector, crossover_point)
        child2_vector = self.create_child_vector(parent2.vector, parent1.vector, crossover_point)

        child1 = Specimen(child1_vector)
        child2 = Specimen(child2_vector)

        # считаем приспособленность особей и записываем значение в поле
        self.specimen_fitness(child1)
        self.specimen_fitness(child2)

        return child1, child2

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

def main():
    print("HI!")
     # Создаем объект алгоритма
    ga = GeneticAlgorithm(population_size = 30, mutation_rate = 0.2, max_number_iterations = 4000)
    
    # Генерируем начальную популяцию
    ga.generate_population()
    initial_fitness, initial_specimen = ga.population_fitness()
    print(f"Начальная приспособленность: {initial_fitness}, лучшая особь: {initial_specimen.vector}")

    for i in range (1, ga.max_number_iterations):
        best_fitness, best_specimen = ga.run_one_iteration()
        if i % 100 == 0:
            print(f"Приспособленность на {i} шаге: {best_fitness}, лучшая особь: {best_specimen.vector}")

if __name__ == "__main__":
    main()