import numpy as np
import pandas as pd


def city_read(file_path, city_size):  # Read cities' coordinates from file
    # Initialization
    coordinate_dict = {}
    csv_data = pd.read_csv(file_path)
    cities = np.array(csv_data)

    # Read start
    x = np.array([cities[i][0] for i in range(0, city_size)])
    y = np.array([cities[i][1] for i in range(0, city_size)])

    for i in range(0, city_size):
        coordinate_dict[i] = (x[i], y[i])
    # Create loop
    coordinate_dict[city_size] = (cities[0][0], cities[0][1])

    return coordinate_dict


def distance_matrix(coordinate_dict, city_size):  # Generate distance matrix
    # Initialization
    dis_mat = np.zeros((city_size + 2, city_size + 2))

    # Distance calculation start
    for i in range(city_size + 1):
        for j in range(city_size + 1):
            if (i != j) and (dis_mat[i][j] == 0):
                x1, y1 = coordinate_dict[i][0], coordinate_dict[i][1]
                x2, y2 = coordinate_dict[j][0], coordinate_dict[j][1]
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                dis_mat[i][j] = dis_mat[j][i] = distance
                # While start point
                if i == 0:
                    dis_mat[city_size + 1][j] = dis_mat[j][city_size + 1] = distance

    return dis_mat


def path_length(dis_mat, path_list, city_size):  # Calculate general path length
    length = 0

    for i in range(city_size):
        length += dis_mat[path_list[i]][path_list[i + 1]]

    return length


def random_individuals(cur_list):  # Generate random individuals
    temp_list = cur_list[1:-1]
    np.random.shuffle(temp_list)
    shuffle_list = cur_list[:1] + temp_list + cur_list[-1:]

    return shuffle_list


def initial_population(city_size, p_num):  # Population initialization
    # Initialization
    path_list = list(range(city_size + 1))
    population = [0] * p_num

    # Generate random path (along with fixed start & end point)
    for i in range(p_num):
        population[i] = random_individuals(path_list)

    return population


def product_len_probability(cur_list, dis_mat, city_size, p_num):  # Calculate current generation's data
    # Initialization
    len_list = []
    prob_list = []
    prob_path = []
    for path in cur_list:
        len_list.append(path_length(dis_mat, path, city_size))

    # Find current situation's best path & length
    max_len = 1e-10 + max(len_list)
    gen_len = min(len_list)
    gen_idx = len_list.index(gen_len)

    # Create list -> Calculate general path length
    mask_list = fitness(p_num, max_len, len_list)
    sum_len = np.sum(mask_list)

    # Calculate Probability
    for i in range(p_num):
        if i == 0:
            prob_list.append(mask_list[i] / sum_len)
        elif i == p_num - 1:
            prob_list.append(1)
        else:
            prob_list.append(prob_list[i - 1] + mask_list[i] / sum_len)

    # Generate path probability
    for i in range(p_num):
        prob_path.append([cur_list[i], len_list[i], prob_list[i]])

    return cur_list[gen_idx], gen_len, prob_path


def fitness(p_num, max_len, len_list):  # Calculate fitness
    fitness_result = np.ones(p_num) * max_len - np.array(len_list)

    return fitness_result


def selection(population, p_num):  # Roulette Wheel Selection
    # Generate aim
    aim = np.random.random()
    if aim < population[0][2]:
        return 0

    # Selection start
    l = 1
    r = p_num

    while l < r:
        mid = int((l + r) / 2)
        if aim > population[mid][2]:
            l = mid
        elif aim < population[mid - 1][2]:
            r = mid
        else:
            return mid


def mutation(cur_list, city_size):  # Mutation
    # Generate mutate position
    pos_1 = np.random.randint(1, city_size + 1)
    pos_2 = np.random.randint(1, city_size + 1)
    while pos_1 == pos_2:
        pos_2 = np.random.randint(1, city_size + 1)

    # Mutation start
    cur_list[pos_1], cur_list[pos_2] = cur_list[pos_2], cur_list[pos_1]

    return cur_list


def cross_over(father, mother, city_size):  # Cross over
    # Initialization
    child = father.copy()
    product_set = np.random.randint(1, city_size + 1)  # set函数会将重复元素剔除，并默认按升序排列
    mother_cross_set = set(mother[1:product_set])

    # Cross over start
    cross_complete = 1
    for i in range(1, city_size + 1):
        if child[i] in mother_cross_set:
            child[i] = mother[cross_complete]
            cross_complete += 1
            if cross_complete > product_set:
                break

    return child


def breed_population(gen_path, gen_len, population, dis_mat, city_size, p_num, iterations_limit, prob_mutation,
                     prob_crossover):
    # Initialization
    child_list = [0] * p_num
    best_path, best_len = gen_path, gen_len
    every_gen_best = [gen_len]

    # Iterative start
    for i in range(iterations_limit):
        child_num = 0

        # Generate Next generation
        while child_num < p_num:

            # Select Parents  -> Produce New individuals
            father = population[selection(population, p_num)][0]
            mother = population[selection(population, p_num)][0]
            child_1, child_2 = father.copy(), mother.copy()

            # Crossover
            cross = (np.random.random() < prob_crossover)
            if cross:
                child_1 = cross_over(father, mother, city_size)
                child_2 = cross_over(mother, father, city_size)

            # Mutate
            # First child
            mutate = (np.random.random() < prob_mutation)
            if mutate:
                child_1 = mutation(child_1, city_size)
            # Update child list
            child_list[child_num] = child_1
            child_num += 1
            if child_num == p_num:
                break

            # Second child
            mutate = (np.random.random() < prob_mutation)
            if mutate:
                child_2 = mutation(child_2, city_size)
            # update child list
            child_list[child_num] = child_2
            child_num += 1

        # Calculate current generation's data
        child_list[0] = gen_path
        gen_path, gen_len, population = product_len_probability(child_list, dis_mat, city_size, p_num)

        # Update overall best length
        if gen_len < best_len:
            best_path = gen_path
            best_len = gen_len

        # Update part best length
        every_gen_best.append(gen_len)

    return best_path, best_len, gen_path, gen_len, every_gen_best
