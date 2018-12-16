import GA
import matplotlib.pyplot as plt
import os

city_size = None  # city numbers
population_size = 100  # population size

prob_mutation = 0.05  # probability of mutation
prob_crossover = 0.6  # probability of crossover
iterations_limit = 100  # Iteration times

coordinate_dict = None  # city coordinates
dis_mat = None  # distance matrix
path_list = None  # travel path
population = None  # list of each generation's population

# result coordinates
res_x = []
res_y = []


def init():  # Initialization
    global city_size, coordinate_dict, dis_mat, path_list, population

    city_size = int(input("Please input city numbers: "))

    coordinate_dict = GA.city_read(os.path.abspath('.') + "/data/tsp.cities.csv", city_size)

    dis_mat = GA.distance_matrix(coordinate_dict, city_size)

    population = GA.initial_population(city_size, population_size)


def visualization():  # Result visualization
    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(every_gen_best)

    plt.subplot(2, 1, 2)
    plt.scatter(res_x, res_y)
    plt.plot(res_x, res_y)

    plt.grid()
    plt.show()


if __name__ == "__main__":
    init()

    # Calculate generation 0's data
    gen_path, gen_len, population = \
        GA.product_len_probability(population, dis_mat, city_size, population_size)

    # Genetic algorithm start
    best_path, best_len, gen_path, gen_len, every_gen_best = \
        GA.breed_population(gen_path, gen_len, population, dis_mat, city_size, population_size, iterations_limit,
                            prob_mutation, prob_crossover)

    # visualize init
    for point in best_path:
        res_x.append(coordinate_dict[point][0])
        res_y.append(coordinate_dict[point][1])
    res_x.append(coordinate_dict[0][0])
    res_y.append(coordinate_dict[0][1])

    visualization()
