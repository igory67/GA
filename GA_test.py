from GA_main import GeneticAlgorithm, Specimen
import random
import numpy as np

# Глобальные параметры для всех тестов
TEST_VECTOR_SIZE = 7
TEST_POPULATION_SIZE = 10
TEST_COST_MATRIX = [
    [1, 12, 19, 21, 8, 7, 15],
    [5, 8, 15, 2, 14, 9, 11],
    [12, 11, 9, 7, 5, 13, 8],
    [18, 16, 13, 3, 9, 6, 12],
    [6, 13, 11, 15, 6, 17, 14],
    [8, 14, 12, 11, 16, 10, 9],
    [7, 9, 13, 8, 12, 15, 11]
]

def print_population(population):
    """Вспомогательная функция для красивого вывода популяции."""
    for i, specimen in enumerate(population):
        print(f"  Особь {i+1}: приспособленность = {specimen.fitness}, вектор = {specimen.vector}")

def setup_ga_instance() -> GeneticAlgorithm:
    """Вспомогательная функция для создания и настройки объекта GA."""
    ga = GeneticAlgorithm(
        population_size=TEST_POPULATION_SIZE, 
        mutation_rate=0.1, 
        max_number_iterations=100, 
        stagnation=0
    )
    ga.vector_size = TEST_VECTOR_SIZE
    ga.cost_matrix = TEST_COST_MATRIX
    return ga

def test_generation_individual():
    print("--- Тест: generation_individual ---")
    ga = setup_ga_instance()
    specimen = ga.generation_individual()
    print(f"Сгенерированная особь: {specimen.vector}")
    print(f"Длина вектора: {len(specimen.vector)}")
    print(f"Уникальные элементы: {len(set(specimen.vector))}")
    print("-" * 30 + "\n")

def test_generate_population():
    print("--- Тест: generate_population ---")
    ga = setup_ga_instance()
    ga.generate_population()
    print(f"Сгенерирована популяция из {len(ga.population)} особей.")
    print_population(ga.population)
    print("-" * 30 + "\n")

def test_specimen_fitness():
    print("--- Тест: specimen_fitness ---")
    ga = setup_ga_instance()
    specimen = Specimen(list(range(TEST_VECTOR_SIZE)), 0)
    fitness = ga.specimen_fitness(specimen)
    print(f"Матрица стоимости:\n" + "\n".join(map(str, ga.cost_matrix)))
    print(f"\nОсобь: {specimen.vector}")
    print(f"Рассчитанная приспособленность: {fitness}")
    print("-" * 30 + "\n")

def test_mutation_specimen():
    print("--- Тест: mutation_specimen ---")
    ga = setup_ga_instance()
    ga.mutation_rate = 1.0
    original_specimen = Specimen(list(range(TEST_VECTOR_SIZE)), 100)
    print(f"Оригинальная особь: {original_specimen.vector}")
    mutated_specimen = ga.mutation_specimen(original_specimen)
    print(f"Мутировавшая особь: {mutated_specimen.vector}")
    print(f"\nПриспособленность оригинальной особи 1: {original_specimen.fitness}")
    print(f"Приспособленность мутировавшей особи 2: {mutated_specimen.fitness}")
    print("-" * 30 + "\n")

def test_crossover():
    print("--- Тест: crossover ---")
    ga = setup_ga_instance()
    parent1 = Specimen(list(range(TEST_VECTOR_SIZE)), 10)
    parent2 = Specimen(list(reversed(range(TEST_VECTOR_SIZE))), 10)
    print(f"Родитель 1: {parent1.vector}")
    print(f"Родитель 2: {parent2.vector}")
    
    child = ga.crossover(parent1, parent2)
    print(f"\nПотомок: {child.vector}")
    print("-" * 30 + "\n")

def test_select_parents_inbreeding():
    print("--- Тест: select_parents_inbreeding ---")
    ga = setup_ga_instance()
    random.seed(42)
    np.random.seed(42)
    ga.generate_population()
    
    print("Тестовая популяция:")
    print_population(ga.population)
    
    parent1_for_test = ga.population[0]
    other_specimens = ga.population[1:]
    
    distances = [ga.hamming_distance(parent1_for_test.vector, s.vector) for s in other_specimens]

    print(f"\nРасстояния Хэмминга от родителя 1 ({parent1_for_test.vector}):")
    for i, dist in enumerate(distances):
        print(f"  до особи {i+2}: {dist}")
    

    original_choice = random.choice
    random.choice = lambda p: p[0]
    parent1_selected, parent2_selected = ga.select_parents_inbreeding()
    random.choice = original_choice
    
    ind = 0
    for i, item in enumerate(other_specimens):
        if item == parent2_selected:
            ind = i + 2
    
    print(f"\nВыбраны: Особь 1 и Особь {ind}")
    print(f"\nВыбранный родитель 1: {parent1_selected.vector}")
    print(f"Выбранный родитель 2: {parent2_selected.vector}")
    print("-" * 30 + "\n")

def test_roulette_selection():
    print("--- Тест: roulette_selection ---")
    ga = setup_ga_instance()
    ga.generate_population()
    print("Начальная популяция:")
    print_population(ga.population)

    ga.population_size = 5
    ga.roulette_selection()
    print("\nПопуляция после отбора рулеткой (топ - 5):")
    print_population(ga.population)
    print("-" * 30 + "\n")

def test_population_fitness():
    print("--- Тест: population_fitness ---")
    ga = setup_ga_instance()
    ga.population = [
        Specimen([0, 1, 2, 3, 4, 5, 6], 0),
        Specimen([6, 5, 4, 3, 2, 1, 0], 0),
        Specimen([1, 2, 3, 4, 5, 6, 0], 0),
        Specimen([2, 3, 4, 5, 6, 0, 1], 0),
        Specimen([3, 4, 5, 6, 0, 1, 2], 0),
        Specimen([4, 5, 6, 0, 1, 2, 3], 0),
        Specimen([5, 6, 0, 1, 2, 3, 4], 0),
        Specimen([6, 0, 1, 2, 3, 4, 5], 0),
        Specimen([0, 2, 1, 3, 4, 5, 6], 0),
        Specimen([1, 0, 2, 3, 4, 6, 5], 0)
    ]
    for item in ga.population:
        ga.specimen_fitness(item)

    print("Тестовая популяция:")
    print_population(ga.population)
    avg_fitness = ga.population_fitness()
    print(f"\nРассчитанная средняя приспособленность: {avg_fitness}")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    test_generation_individual()
    test_generate_population()
    test_specimen_fitness()
    test_population_fitness()
    test_mutation_specimen()
    test_crossover()
    test_select_parents_inbreeding()
    test_roulette_selection()
    test_population_fitness()