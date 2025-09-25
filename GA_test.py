from GA_main import GeneticAlgorithm, Specimen
import random
import numpy as np

def print_population(population):
    """Вспомогательная функция для красивого вывода популяции."""
    for i, specimen in enumerate(population):
        print(f"  Особь {i+1}: fitness={specimen.fitness}, vector={specimen.vector}")

def test_generation_individual():
    print("--- Тест: generation_individual ---")
    ga = GeneticAlgorithm(population_size=5, mutation_rate=0.1, max_number_iterations=100, stagnation=0)
    specimen = ga.generation_individual()
    print(f"Сгенерированная особь: {specimen.vector}")
    print(f"Длина вектора: {len(specimen.vector)}")
    print(f"Уникальные элементы: {len(set(specimen.vector))}")
    print("Тест пройден, если длина вектора равна vector_size и все элементы уникальны.")
    print("-" * 30 + "\n")

def test_generate_population():
    print("--- Тест: generate_population ---")
    ga = GeneticAlgorithm(population_size=5, mutation_rate=0.1, max_number_iterations=100, stagnation=0)
    ga.generate_population()
    print(f"Сгенерирована популяция из {len(ga.population)} особей.")
    print_population(ga.population)
    print("Тест пройден, если сгенерировано N особей с рассчитанной приспособленностью.")
    print("-" * 30 + "\n")

def test_is_acceptable():
    print("--- Тест: is_acceptable ---")
    ga = GeneticAlgorithm(population_size=5, mutation_rate=0.1, max_number_iterations=100, stagnation=0)
    valid_specimen = Specimen([0, 1, 2, 3, 4], 0)
    invalid_specimen = Specimen([0, 1, 2, 1, 4], 0)
    print(f"Проверка допустимой особи: {ga.is_acceptable(valid_specimen)}")
    print(f"Проверка недопустимой особи: {ga.is_acceptable(invalid_specimen)}")
    print("Тест пройден, если для первой особи результат True, а для второй False.")
    print("-" * 30 + "\n")

def test_specimen_fitness():
    print("--- Тест: specimen_fitness ---")
    ga = GeneticAlgorithm(population_size=5, mutation_rate=0.1, max_number_iterations=100, stagnation=0)
    ga.vector_size = 5
    ga.cost_matrix = [
        [10, 12, 19, 21, 8],
        [5, 10, 15, 2, 14],
        [12, 11, 9, 7, 5],
        [18, 16, 13, 10, 9],
        [6, 13, 11, 15, 10]
    ]
    specimen = Specimen([0, 1, 2, 3, 4], 0)
    fitness = ga.specimen_fitness(specimen)
    # Ожидаемый результат: 10 + 10 + 9 + 10 + 10 = 49
    print(f"Матрица стоимости:\n{ga.cost_matrix}")
    print(f"Особь: {specimen.vector}")
    print(f"Рассчитанная приспособленность: {fitness}")
    print(f"Ожидаемая приспособленность: 49")
    print("Тест пройден, если рассчитанное значение совпадает с ожидаемым.")
    print("-" * 30 + "\n")

def test_find_best_specimen():
    print("--- Тест: find_best_specimen (population_fitness) ---")
    ga = GeneticAlgorithm(population_size=5, mutation_rate=0.1, max_number_iterations=100, stagnation=0)
    ga.vector_size = 5
    ga.population = [
        Specimen([0, 1, 2, 3, 4], 50),
        Specimen([4, 3, 2, 1, 0], 25),
        Specimen([1, 2, 3, 4, 0], 35),
        Specimen([3, 2, 1, 0, 4], 28),
        Specimen([2, 0, 1, 4, 3], 60)
    ]
    print("Тестовая популяция:")
    print_population(ga.population)
    best_fitness, best_specimen = ga.find_best_specimen()
    print(f"Найденная лучшая приспособленность: {best_fitness}")
    print(f"Найденная лучшая особь: {best_specimen.vector}")
    print("Ожидаемая лучшая приспособленность: 25")
    print("Тест пройден, если найденная особь имеет наименьшую приспособленность.")
    print("-" * 30 + "\n")

def test_mutation_specimen():
    print("--- Тест: mutation_specimen ---")
    ga = GeneticAlgorithm(population_size=5, mutation_rate=1.0, max_number_iterations=100, stagnation=0) # 100% мутация
    original_specimen = Specimen(list(range(ga.vector_size)), 100)
    print(f"Оригинальная особь: {original_specimen.vector}")
    mutated_specimen = ga.mutation_specimen(original_specimen)
    print(f"Мутировавшая особь: {mutated_specimen.vector}")
    print(f"Приспособленность мутанта: {mutated_specimen.fitness}")
    print("Тест пройден, если вектор мутанта отличается от оригинала, но содержит те же гены.")
    print("-" * 30 + "\n")

def test_shaping_next_generation():
    print("--- Тест: shaping_next_generation ---")
    ga = GeneticAlgorithm(population_size=4, mutation_rate=0.1, max_number_iterations=100, stagnation=0)
    ga.vector_size = 5
    parents = [
        Specimen([0, 1, 2, 3, 4], 50),
        Specimen([4, 3, 2, 1, 0], 25),
        Specimen([1, 2, 3, 4, 0], 35),
        Specimen([3, 2, 1, 0, 4], 28)
    ]
    offspring = [
        Specimen([2, 0, 1, 4, 3], 60),
        Specimen([1, 0, 4, 2, 3], 15),
        Specimen([4, 1, 0, 3, 2], 22),
        Specimen([0, 3, 2, 1, 4], 45)
    ]
    print("Родители:")
    print_population(parents)
    print("Потомки:")
    print_population(offspring)
    
    next_gen = ga.shaping_next_generation(parents, offspring)
    print("\nСформированное следующее поколение (топ-4):")
    print_population(next_gen)
    print("Ожидаемые fitness'ы в новом поколении: 15, 22, 25, 28")
    print("Тест пройден, если в новом поколении 4 особи с наилучшей (минимальной) приспособленностью.")
    print("-" * 30 + "\n")

def test_crossover():
    print("--- Тест: crossover ---")
    ga = GeneticAlgorithm(population_size=5, mutation_rate=0.1, max_number_iterations=100, stagnation=0)
    ga.vector_size = 5
    parent1 = Specimen([0, 1, 2, 3, 4], 10)
    parent2 = Specimen([4, 3, 2, 1, 0], 10)
    print(f"Родитель 1: {parent1.vector}")
    print(f"Родитель 2: {parent2.vector}")
    # Для наглядности установим точку кроссовера вручную
    random.seed(0) # делает результат randint предсказуемым
    crossover_point = random.randint(1, ga.vector_size - 2)
    print(f"Точка кроссовера: {crossover_point}")
    
    child = ga.crossover(parent1, parent2)
    print(f"Потомок: {child.vector}, fitness={child.fitness}")
    print("Тест пройден, если потомки являются корректными перестановками.")
    print("-" * 30 + "\n")

def test_select_parents_inbreeding():
    print("--- Тест: select_parents_inbreeding ---")
    ga = GeneticAlgorithm(population_size=4, mutation_rate=0.1, max_number_iterations=100, stagnation=0)
    ga.vector_size = 10
    
    # Задаем seed для воспроизводимости результатов
    random.seed(42) 

    # Создаем более разнообразные векторы
    p1_vec = np.random.permutation(10).tolist()
    p2_vec = np.random.permutation(10).tolist()
    p3_vec = np.random.permutation(10).tolist()
    p4_vec = np.random.permutation(10).tolist()

    ga.population = [
        Specimen(p1_vec, 10),
        Specimen(p2_vec, 5),
        Specimen(p3_vec, 15),
        Specimen(p4_vec, 8)
    ]
    
    # Чтобы тест был предсказуемым, выберем первого родителя вручную
    # random.seed(0) # random.choice выберет первого
    
    print("Тестовая популяция:")
    print_population(ga.population)
    
    # Вычисляем расстояния для наглядности
    parent1 = ga.population[0] # Берем первого для предсказуемости
    print(f"\nРасстояния Хэмминга от родителя 1 ({parent1.vector}):")
    for i, spec in enumerate(ga.population[1:]):
        dist = ga.hamming_distance(parent1.vector, spec.vector)
        print(f"  до особи {i+2}: {dist}")

    # Выполняем сам выбор
    parent1_selected, parent2_selected = ga.select_parents_inbreeding()
    
    print(f"\nВыбранный родитель 1: {parent1_selected.vector}")
    print(f"Выбранный родитель 2: {parent2_selected.vector}")
    print("Тест пройден, если второй родитель имеет минимальное расстояние Хэмминга до первого.")
    print("-" * 30 + "\n")

def test_roulette_selection():
    print("--- Тест: roulette_selection ---")
    ga = GeneticAlgorithm(population_size = 10, mutation_rate = 0.2, max_number_iterations = 500, stagnation = 0)
    
    ga.generate_population()
    print("Начальная популяция:")
    print_population(ga.population)

    ga.roulette_selection()
    print("\nПопуляция после отбора рулеткой:")
    print_population(ga.population)
    print("Тест пройден, если новая популяция сформирована (размер может быть меньше).")
    print("-" * 30 + "\n")

def test_population_fitness():
    print("--- Тест: population_fitness ---")
    ga = GeneticAlgorithm(population_size=4, mutation_rate=0.1, max_number_iterations=100, stagnation=0)
    ga.population = [
        Specimen([0, 1, 2, 3], 10),
        Specimen([0, 1, 3, 2], 20),
        Specimen([3, 2, 1, 0], 30),
        Specimen([0, 2, 1, 3], 40)
    ]
    print("Тестовая популяция:")
    print_population(ga.population)
    
    avg_fitness = ga.population_fitness()
    expected_avg = (10 + 20 + 30 + 40) / 4
    
    print(f"\nРассчитанная средняя приспособленность: {avg_fitness}")
    print(f"Ожидаемая средняя приспособленность: {expected_avg}")
    print("Тест пройден, если рассчитанное значение совпадает с ожидаемым.")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    test_generation_individual()
    test_generate_population()
    test_is_acceptable()
    test_specimen_fitness()
    test_find_best_specimen()
    test_population_fitness()
    test_mutation_specimen()
    test_shaping_next_generation()
    test_crossover()
    test_select_parents_inbreeding()
    test_roulette_selection()
    test_population_fitness()
