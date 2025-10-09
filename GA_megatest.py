import matplotlib.pyplot as plt
import time
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
from GA_main_newest import GeneticAlgorithm, Specimen  

def example_function(ga: GeneticAlgorithm, num_iterations: int) -> Tuple[List[float], List[Specimen], float]:
    """
    Запускает генетический алгоритм на заданное количество итераций и собирает статистику.
    
    Parameters:
    ga: GeneticAlgorithm - объект генетического алгоритма
    num_iterations: int - количество итераций для выполнения
    
    Returns:
    Tuple[List[float], List[Specimen], float] - 
        - список лучших приспособленностей на каждой итерации
        - список лучших особей на каждой итерации
        - время выполнения в секундах
    """
    best_fitness_history = []
    best_specimen_history = []
    
    start_time = time.time()
    
    best_fitness, best_specimen = ga.run_one_iteration()
    current_spic_fitness = best_fitness
    current_specimen = best_specimen

    best_fitness_history.append(current_spic_fitness)
    best_specimen_history.append(current_specimen)
    last_fitness = ga.population_fitness()
    
    for i in range(1, num_iterations):
        current_spic_fitness, current_specimen = ga.run_one_iteration()

        current_fitness = ga.population_fitness()


        if abs(current_fitness - last_fitness) < ga.stagnation:
            print(f"Итерация {i + 1}: Лучшая приспособленность = {best_fitness}")
            print(f"Лучшая особь: {best_specimen.vector}")
            print("-" * 50)
            break

        if min(current_spic_fitness, best_fitness) == current_fitness:
            best_fitness = current_spic_fitness
            best_specimen = current_specimen

        last_fitness = current_fitness

        best_fitness_history.append(current_spic_fitness)
        best_specimen_history.append(current_specimen)
    
    # Вывод информации каждые N итераций
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Итерация {i + 1}: Лучшая приспособленность = {best_fitness}")
            print(f"Лучшая особь: {best_specimen.vector}")
            print("-" * 50)

    execution_time = time.time() - start_time
    
    return best_fitness_history, best_specimen_history, execution_time

def plot_convergence(best_fitness_history: List[float], population_size: int, 
                    mutation_rate: float, num_iterations: int, minim_fitness : int):
    """
    Строит график сходимости алгоритма.
    
    Parameters:
    best_fitness_history: List[float] - история лучших приспособленностей
    population_size: int - размер популяции
    mutation_rate: float - вероятность мутации
    num_iterations: int - количество итераций
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(best_fitness_history) + 1), best_fitness_history, 
             linewidth=2, color='blue', alpha=0.7)
    plt.xlabel('Номер итерации')
    plt.ylabel('Лучшая приспособленность')
    plt.title(f'Сходимость ГА (N={population_size}, p_m={mutation_rate})')
    plt.grid(True, alpha=0.3)
    
    # Добавляем аннотацию с финальным результатом
    final_fitness = minim_fitness
    plt.annotate(f'Финальный результат: {final_fitness:.2f}', 
                xy=(0.7, 0.1), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_performance_comparison(performance_data: Dict[int, float]):
    """
    Строит график сравнения времени выполнения для разных размеров популяции.
    
    Parameters:
    performance_data: Dict[int, float] - словарь {размер_популяции: время_выполнения}
    """
    sizes = list(performance_data.keys())
    times = list(performance_data.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, color='red')
    plt.xlabel('Размер популяции (N)')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Зависимость времени выполнения от размера популяции')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # Добавляем аннотации с временем для каждой точки
    for size, time_val in zip(sizes, times):
        plt.annotate(f'{time_val:.3f}с', 
                    xy=(size, time_val), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_final_fitness_comparison(fitness_data: Dict[int, float]):
    """
    Строит график сравнения финальной приспособленности для разных размеров популяции.
    
    Parameters:
    fitness_data: Dict[int, float] - словарь {размер_популяции: финальная_приспособленность}
    """
    sizes = list(fitness_data.keys())
    fitness_values = list(fitness_data.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(sizes, fitness_values, 's-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Размер популяции (N)')
    plt.ylabel('Финальная приспособленность')
    plt.title('Зависимость финальной приспособленности от размера популяции')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Инвертируем ось Y (меньшие значения лучше)
    plt.gca().invert_yaxis()
    
    # Добавляем аннотации с приспособленностью для каждой точки
    for size, fitness_val in zip(sizes, fitness_values):
        plt.annotate(f'{fitness_val:.1f}', 
                    xy=(size, fitness_val), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)
    
    plt.tight_layout()
    plt.show()

def run_comprehensive_analysis():
    """
    Проводит комплексный анализ алгоритма для разных размеров популяции.
    """
    # Параметры тестирования
    population_sizes = [100, 150, 200]#, 200, 300]
    # population_sizes = [2, 20, 30, 100]#, 200, 300]
    number_of_objects = 10#0
    mutation_rate = 0.15#0.2
    num_iterations = 400#200
    stagnation = 0.001
    
    performance_results = {}
    fitness_results = {}
    convergence_data = {}
    
    print("=" * 70)
    print("КОМПЛЕКСНЫЙ АНАЛИЗ ГЕНЕТИЧЕСКОГО АЛГОРИТМА")
    print("=" * 70)
    

    for pop_size in population_sizes:
        print(f"\nТестирование для N = {pop_size}")
        print("-" * 40)
        
        minimum = 2**64
        specimenn = []

        #объект
        ga = GeneticAlgorithm(population_size=pop_size,mutation_rate=mutation_rate,number_of_objects=number_of_objects,max_number_iterations=num_iterations,stagnation=stagnation)
        print(ga.cost_matrix)
        #алгоритм
        best_fitness_history, best_specimen_history, execution_time = example_function(ga, num_iterations)
        
        #результаты в соответствующие массивы
        performance_results[pop_size] = execution_time
        fitness_results[pop_size] = best_fitness_history[-1]
        convergence_data[pop_size] = best_fitness_history

        if min(best_fitness_history[-1], minimum) == best_fitness_history[-1]:
            specimenn = best_specimen_history[-1].vector
            minimum = best_fitness_history[-1]
            

        print(f"Размер популяции: {pop_size}")
        print(f"Время выполнения: {execution_time:.3f} секунд")
        print(f"Финальная приспособленность: {minimum}")
        print(f"Лучшая особь: {specimenn}")
        
        # Строим график сходимости для каждого размера популяции
        plot_convergence(best_fitness_history, pop_size, mutation_rate, num_iterations, minimum)
    
    # Строим сравнительные графики
    print("\n" + "=" * 70)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 70)
    
    #plot_performance_comparison(performance_results)
    #plot_final_fitness_comparison(fitness_results)
    
    # Строим комбинированный график сходимости для нескольких размеров популяции
    plot_multiple_convergence(convergence_data, mutation_rate)
    
    # Выводим таблицу результатов
    #print_results_table(performance_results, fitness_results)

def plot_multiple_convergence(convergence_data: Dict[int, List[float]], mutation_rate: float):
    """
    Строит график сходимости для нескольких размеров популяции на одном графике.
    
    Parameters:
    convergence_data: Dict[int, List[float]] - словарь с историями приспособленностей
    mutation_rate: float - вероятность мутации
    """
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(convergence_data)))
    
    for i, (pop_size, fitness_history) in enumerate(convergence_data.items()):
        iterations = range(1, len(fitness_history) + 1)
        plt.plot(iterations, fitness_history, 
                label=f'N={pop_size}', 
                linewidth=2, 
                alpha=0.7,
                color=colors[i])
    
    plt.xlabel('Номер итерации')
    plt.ylabel('Лучшая приспособленность')
    plt.title(f'Сравнение сходимости для разных размеров популяции (p_m={mutation_rate})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def print_results_table(performance_results: Dict[int, float], fitness_results: Dict[int, float]):
    """
    Выводит таблицу с результатами тестирования.
    
    Parameters:
    performance_results: Dict[int, float] - результаты по времени
    fitness_results: Dict[int, float] - результаты по приспособленности
    """
    print("\nТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    print(f"{'Размер популяции':<15} {'Время (с)':<12} {'Приспособленность':<18}")
    print("-" * 60)
    
    for pop_size in performance_results.keys():
        time_val = performance_results[pop_size]
        fitness_val = fitness_results[pop_size]
        print(f"{pop_size:<15} {time_val:<12.3f} {fitness_val:<18.2f}")
    
    print("=" * 60)





if __name__ == "__main__":

    run_comprehensive_analysis()