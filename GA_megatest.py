import matplotlib.pyplot as plt
import time
import numpy as np
from typing import List, Tuple, Dict
from GA_main import GeneticAlgorithm, Specimen  

def example_function(ga: GeneticAlgorithm, num_iterations: int, stagnation_max_val) -> Tuple[List[float], List[Specimen], float]:
    """
    Запускает генетический алгоритм на заданное количество итераций и собирает статистику.
    
    Parameters:
    ga: GeneticAlgorithm - объект генетического алгоритма
    num_iterations: int - количество итераций для выполнения
    
    Returns:
        - список лучших приспособленностей на каждой итерации
        - список лучших особей на каждой итерации
        - время выполнения в секундах
        - avg_fitness_history
    """
    best_fitness_history = []
    best_specimen_history = []
    avg_fitness_history = []
    
    start_time = time.time()
    
    best_fitness, best_specimen = ga.run_one_iteration()
    current_spic_fitness = best_fitness
    current_specimen = best_specimen

    best_fitness_history.append(current_spic_fitness)
    best_specimen_history.append(current_specimen)
    last_fitness = ga.population_fitness()
    avg_fitness_history.append(ga.population_fitness())

    stagnation_counter = 0

    for i in range(1, num_iterations):
        local_best_fitness, local_current_specimen = ga.run_one_iteration()
        

        current_fitness = ga.population_fitness()

        if abs(current_fitness - last_fitness) < ga.stagnation:
            stagnation_counter += 1
            if (stagnation_counter > 2):
                print("stagnation ", stagnation_counter, "best ", local_best_fitness)

        else:
            stagnation_counter = 0            

        if stagnation_counter >= stagnation_max_val:
            print(f"Итерация {i + 1}: Лучшая приспособленность = {best_fitness}")
            print(f"Лучшая особь: {current_specimen.vector}")
            print("-" * 50)
            break

        if local_best_fitness < best_fitness:
            best_fitness = local_best_fitness
            best_specimen = local_current_specimen


        last_fitness = current_fitness
        avg_fitness_history.append(current_fitness)

        best_fitness_history.append(local_best_fitness)
        best_specimen_history.append(local_current_specimen)
    
    # Вывод информации каждые скок-то итераций
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Итерация {i + 1}: Лучшая приспособленность = {local_best_fitness}, avg = {current_fitness}")
            print(f"Лучшая особь: {best_specimen.vector}", "итерация ", i)
            #print("-" * 50)
        iteration_counter = i

    execution_time = time.time() - start_time
    
    return best_fitness_history, best_specimen_history, execution_time,avg_fitness_history,iteration_counter

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

def plot_average_fitness(avg_fitness_history: List[float], population_size: int, 
                         mutation_rate: float, num_iterations: int):
    """
    Строит график изменения средней приспособленности популяции.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(avg_fitness_history) + 1), avg_fitness_history,
             linewidth=2, color='orange', alpha=0.8)
    plt.xlabel('Номер итерации')
    plt.ylabel('Средняя приспособленность популяции')
    plt.title(f'Изменение средней приспособленности (N={population_size}, p_m={mutation_rate})')
    plt.grid(True, alpha=0.3)
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
    population_sizes = [30,40,50,60,70,80,90]#[40,50,70,80,90,100]#[20,50,70,100]#[20, 30, 50, 70, 90, 100]#, 200, 300]
    # population_sizes = [2, 20, 30, 100]#, 200, 300]
    number_of_objects = 18#20#0
    mutation_rate = 0.2#0.15#0.2
    num_iterations = 7500#200

    stagnation = 0.1
    stagnation_max_val = 3
    
    performance_results = {}
    fitness_results = {}
    convergence_data = {}
    execution_times = []
    pop_sisez = []
    print("=" * 70)
    print("АНАЛИЗ ГЕНЕТИЧЕСКОГО АЛГОР0ИТМА")
    

    for pop_size in population_sizes:
        print(f"\nТестирование для N = {pop_size}")
        #print("-" * 40)
        
        minimum = 2**64
        specimenn = []

        #объект
        ga = GeneticAlgorithm(population_size=pop_size,mutation_rate=mutation_rate,number_of_objects=number_of_objects,max_number_iterations=num_iterations,stagnation=stagnation)
        #print(ga.cost_matrix)
        #алгоритм
        best_fitness_history, best_specimen_history, execution_time,avg_fitness_history,iteration_counter = example_function(ga, num_iterations,stagnation_max_val)
        #avg_fitness_history = []
        #результаты в соответствующие массивы
        performance_results[pop_size] = execution_time
        fitness_results[pop_size] = best_fitness_history[-1]
        convergence_data[pop_size] = best_fitness_history

        if min(best_fitness_history[-1], minimum) == best_fitness_history[-1]:
            specimenn = best_specimen_history[-1].vector
            minimum = best_fitness_history[-1]
            

        print(f"Размер популяции: {pop_size}")
        pop_sisez.append(pop_size)
        print(f"Время выполнения: {execution_time:.3f} секунд")
        execution_times.append(execution_time)
        print(f"Финальная приспособленность: {minimum}")
        print(f"лучшая приспособленность: {min(best_fitness_history)}")
        print(f"Лучшая особь: {specimenn}")
        print("Число итераций: ", iteration_counter)
        
        # Строим график сходимости для каждого размера популяции
        plot_fitness_dynamics(best_fitness_history, avg_fitness_history, pop_size, mutation_rate, num_iterations, minimum, execution_time, stagnation, stagnation_max_val, number_of_objects)

        # plot_convergence(best_fitness_history, pop_size, mutation_rate, num_iterations, minimum)
        # plot_average_fitness(avg_fitness_history, pop_size, mutation_rate, num_iterations)

    
    # Строим сравнительные графики
    print("\n" + "=" * 70)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 70)
#общиеграфики
    plot_performance_comparison(performance_results)
    plot_final_fitness_comparison(fitness_results)
    
    # Строим комбинированный график сходимости для нескольких размеров популяции
    plot_multiple_convergence(convergence_data, mutation_rate,execution_time, stagnation, stagnation_max_val,number_of_objects)
    print("Время для разных: ")
    for size, time_val in zip(pop_sisez, execution_times):
        print("размер ", size, "время ", time_val)

    # Выводим таблицу результатов
    print_results_table(performance_results, fitness_results)

def plot_multiple_convergence(convergence_data: Dict[int, List[float]], mutation_rate: float, execution_time, stagnation, stagnation_max_val,number_of_objects):
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
    plt.title(f'Сравнение сходимости для разных размеров популяции (p_m={mutation_rate} t={execution_time}, epsilon={stagnation}, stagn={stagnation_max_val},n={number_of_objects} )')
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


def plot_fitness_dynamics(best_fitness_history: List[float], avg_fitness_history: List[float],
                          population_size: int, mutation_rate: float, num_iterations: int,
                          minim_fitness: float, execution_time, stagnation, stagnation_max_val,number_of_objects):
    """
    Строит график динамики лучшей и средней приспособленности на одной оси.
    """
    plt.figure(figsize=(12, 6))
    
    iterations = range(1, len(best_fitness_history) + 1)
    
    # Лучшая приспособленность
    plt.plot(iterations, best_fitness_history, label='Лучшая приспособленность',
             color='blue', linewidth=2, alpha=0.8)
    
    # Средняя приспособленность
    plt.plot(iterations, avg_fitness_history, label='Средняя приспособленность',
             color='orange', linewidth=2, alpha=0.8)
    
    plt.xlabel('Номер итерации')
    plt.ylabel('Приспособленность')
    plt.title(f'Динамика N={population_size}, p_m={mutation_rate}, i={num_iterations}, '
        f't={execution_time:.3f}s, ε={stagnation}, stagn={stagnation_max_val}, n={number_of_objects}'
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Отметим финальное значение лучшей приспособленности
    plt.annotate(f'Финал: {minim_fitness:.2f}', xy=(len(best_fitness_history), best_fitness_history[-1]),
                 xytext=(-80, 20), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comprehensive_analysis()