# extended
# intermediate
# crossover

import numpy as np

from encoder.tapotl import TLEncoding
from my_utils.logger import Logger
from my_utils.timer import Timer
import pickle

from optimizer.sa_standard import Solution2


def save_pickle(object_pi, path):
    file_pi = open(path, 'wb')
    pickle.dump(object_pi, file_pi)


def load_pickle(path):
    filehandler = open(path, 'rb')
    object = pickle.load(filehandler)
    return object


def get_neighbour_int(x, a):
    r = np.random.randint(-a, a, (1, len(x)))[0]
    return x + r


def get_neighbour_real(x, b):
    r = np.random.uniform(-b, b, (1, len(x)))[0]
    return x + r


def get_neighbour(x, num_neuron=15, activation=1, trainable_layer=3, ei_num_neuron=3, ei_activation=6):
    neighbour_num_neurons = get_neighbour_int(x[:ei_num_neuron], num_neuron)
    neighbour_activation = get_neighbour_real(x[ei_num_neuron:ei_activation], activation)
    neighbour_trainable_layer = get_neighbour_real(x[ei_activation:], trainable_layer)
    return np.array([*neighbour_num_neurons, *neighbour_activation, *neighbour_trainable_layer])


# extended intermediate crossover
def eicrossover(p1, p2):
    a = np.random.uniform(-0.25, 1.25, (1, len(p1)))[0]
    c1 = p1 + a * (p2 - p1)
    c2 = p2 + a * (p1 - p2)
    return c1, c2


def mutate(x,
           threshold_mutation,
           optimized_num_layer,
           len_encoding,
           lb,
           ub,
           log_message=True):
    is_mutated = False
    rand_pos, rand_sl = np.random.rand(), np.random.rand()
    next_p = np.copy(x)
    if rand_pos < threshold_mutation or rand_sl < threshold_mutation:
        is_mutated = True

        if log_message:
            print("UNIFORM MUTATION")

        if rand_pos < threshold_mutation:
            inew = np.random.randint(0, len_encoding)
            num_new = int(np.round(np.random.uniform(lb[inew], ub[inew], (1))[0], 0))
            np.put(next_p, [inew], num_new)

            if log_message:
                print("Mutation position")

        if rand_sl < threshold_mutation:
            inew = np.random.randint(0, optimized_num_layer)
            num_new = 0 if next_p[inew] > 0 else np.random.randint(1, optimized_num_layer)

            np.put(next_p, [inew], num_new)

            if log_message:
                print("Flip Layer")

    return is_mutated, next_p


# np.log is ln in numpy, whereas np.log10 is base 10 log
def boltzmann_annealing(T0, i):
    # print("i inside boltzman ennaling")
    # print(np.log(1 + i))
    return T0 / np.log(1 + i)

def EVOSA(max_time,
          ttype="iters",
          Objective=None,
          T0=0.3,
          Encoder=TLEncoding,
          threshold_mutation=0.2,
          lb=None, ub=None, log_message=True, log_stored_path="SA_topology.csv",
          best_stored_path="EvoSA.obj",
          evo_only=False,
          start_encoding=None):
    log_stored_path = f"output/logs/{log_stored_path}"

    if start_encoding is None:
        fc = Encoder()
    else:
        print("Start Encoding...")
        fc = Encoder(start_encoding)
        print(f"Encoding: {fc}")

    res = Objective(fc.encoding)
    current_solution = Solution2(fc.encoding,
                                 res['cost'],
                                 res['cost_train'],
                                 res['cost_val'],
                                 res['cost_test'],
                                 res['cost_top5_train'],
                                 res['cost_top5_val'],
                                 res['cost_top5_test'],
                                 fc.key_str(),
                                 res['total_params'],
                                 res['latency'])

    best_solution = current_solution.clone()

    if evo_only:
        res = Objective(fc.encoding)
        current_solution = Solution2(fc.encoding,
                                     res['cost'],
                                     res['cost_train'],
                                     res['cost_val'],
                                     res['cost_test'],
                                     res['cost_top5_train'],
                                     res['cost_top5_val'],
                                     res['cost_top5_test'],
                                     fc.key_str(),
                                     res['total_params'],
                                     res['latency'])

        best_solution = current_solution.clone()

    if log_message:
        print("First solution: ", current_solution)

    t = Timer(max_time=max_time, ttype=ttype)
    T_curr = T0
    log = Logger()

    optimized_num_layer = fc.optimized_num_layers()
    len_encoding = len(fc.encoding)
    if not evo_only:
        while t.run():
                # print(f"p: {current_solution.p}")
                is_mutate, next_p = mutate(current_solution.p,
                                           threshold_mutation,
                                           optimized_num_layer,
                                           len_encoding,
                                           lb,
                                           ub,
                                           log_message=True)
                if is_mutate:
                    note = 'Mutate'
                else:
                    note = 'Normal'
                    next_p = get_neighbour(current_solution.p)
                    next_p = np.clip(next_p, lb, ub)

                # print(f"next_p: {next_p}")

                res = Objective(next_p)
                next_solution = Solution2(next_p,
                                          res['cost'],
                                          res['cost_train'],
                                          res['cost_val'],
                                          res['cost_test'],
                                          res['cost_top5_train'],
                                          res['cost_top5_val'],
                                          res['cost_top5_test'],
                                          Encoder(next_p).key_str(),
                                          res['total_params'],
                                          res['latency'])

                if log_message:
                    print(f"{t.current_time + 1} -> next candidate solution: {next_solution}")

                # T_curr = update_temp(T_curr, t.current_time, max_time)
                T_curr = boltzmann_annealing(T0, t.current_time + 1)

                if next_solution.cost < current_solution.cost:
                    current_solution.copy(next_solution)

                    if log_message:
                        print(f"IMPROVED ! {note}")

                    note += ' Better'

                    if next_solution.cost < best_solution.cost:
                        best_solution.copy(next_solution)
                        note += ' BEST'

                # elif np.exp((current_solution.cost - next_solution.cost) / T_curr) > np.random.rand() and (
                #         next_solution.cost - current_solution.cost) < 0.2:
                elif np.exp((current_solution.cost - next_solution.cost) / T_curr) > np.random.rand():
                    current_solution.copy(next_solution)
                    note += ' Accept Bad'
                    print("Accept bad solution -->")

                if log_message:
                    print("current solution: ", current_solution)

                log.add("NOID", t.current_time + 1, current_solution.p, current_solution.cost, current_solution.cost_train,
                        current_solution.cost_val, current_solution.total_params, current_solution.latency, note)
                log.save_csv(log_stored_path)
                t.next()

    # 10 + 40 (4*10)
    print("CROSS OVER MODE")
    parents = []
    parents_cost = []
    parents_solution = []
    parents.append(best_solution.p)
    parents_cost.append(best_solution.cost)
    parents_solution.append(best_solution)
    # 3 means 3 new parents as initial population (tot 4 with best solution)
    for i in range(3):
        p_parent = Encoder()
        c1, c2 = eicrossover(best_solution.p, p_parent.encoding)
        c1 = np.clip(c1, lb, ub)
        c2 = np.clip(c2, lb, ub)
        res_c1 = Objective(c1)
        res_c2 = Objective(c2)

        c1_solution = Solution2(c1,
                                res_c1['cost'],
                                res_c1['cost_train'],
                                res_c1['cost_val'],
                                res_c1['cost_test'],
                                res_c1['cost_top5_train'],
                                res_c1['cost_top5_val'],
                                res_c1['cost_top5_test'],
                                Encoder(c1).key_str(),
                                res_c1['total_params'],
                                res_c1['latency'])

        c2_solution = Solution2(c2,
                                res_c2['cost'],
                                res_c2['cost_train'],
                                res_c2['cost_val'],
                                res_c2['cost_test'],
                                res_c2['cost_top5_train'],
                                res_c2['cost_top5_val'],
                                res_c2['cost_top5_test'],
                                Encoder(c1).key_str(),
                                res_c2['total_params'],
                                res_c2['latency'])

        if res_c1['cost'] < res_c2['cost']:
            parents.append(c1)
            parents_cost.append(res_c1['cost'])
            parents_solution.append(c1_solution)
        else:
            parents.append(c2)
            parents_cost.append(res_c2['cost'])
            parents_solution.append(c2_solution)

    for i in range(10):
        children = []
        children_cost = []
        children_solution = []
        for i in range(2):
            p_indices = np.arange(4)
            np.random.shuffle(p_indices)
            p1_index = p_indices[0]
            p2_index = p_indices[1]

            c1, c2 = eicrossover(parents[p1_index], parents[p2_index])
            c1 = np.clip(c1, lb, ub)
            c2 = np.clip(c2, lb, ub)
            res_c1 = Objective(c1)
            res_c2 = Objective(c2)

            c1_solution = Solution2(c1,
                                    res_c1['cost'],
                                    res_c1['cost_train'],
                                    res_c1['cost_val'],
                                    res_c1['cost_test'],
                                    res_c1['cost_top5_train'],
                                    res_c1['cost_top5_val'],
                                    res_c1['cost_top5_test'],
                                    Encoder(c1).key_str(),
                                    res_c1['total_params'],
                                    res_c1['latency'])

            c2_solution = Solution2(c2,
                                    res_c2['cost'],
                                    res_c2['cost_train'],
                                    res_c2['cost_val'],
                                    res_c2['cost_test'],
                                    res_c2['cost_top5_train'],
                                    res_c2['cost_top5_val'],
                                    res_c2['cost_top5_test'],
                                    Encoder(c2).key_str(),
                                    res_c2['total_params'],
                                    res_c2['latency'])

            children.append(c1)
            children.append(c2)
            children_cost.append(res_c1['cost'])
            children_cost.append(res_c2['cost'])
            children_solution.append(c1_solution)
            children_solution.append(c2_solution)

        #     Combining parents and children
        temp_pop = children + parents
        temp_pop_cost = children_cost + parents_cost
        temp_pop_solution = children_solution + parents_solution
        isorted = np.argsort(np.array(temp_pop_cost))
        parents = [temp_pop[e] for i, e in enumerate(isorted[:4])]
        parents_cost = [temp_pop_cost[e] for i, e in enumerate(isorted[:4])]
        parents_solution = [temp_pop_solution[e] for i, e in enumerate(isorted[:4])]
        if log_message:
            print("best solution so far from crossoving")
            print(f"Best solution: {parents_solution[0]}")

        log.add("NOID", t.current_time + i + 1, parents_solution[0].p, parents_solution[0].cost, parents_solution[0].cost_train,
                parents_solution[0].cost_val, parents_solution[0].total_params, parents_solution[0].latency, "Best crossover elitism")
        log.save_csv(log_stored_path)
    best_solution = parents_solution[0]
    save_pickle(best_solution, f"output/saved_objects/{best_stored_path}")
    return best_solution

# x = np.array([9,8,7,6])
# # iss = np.argsort(x)
# # print(x[iss[:2]])
#
# a = [0, 3]
# print(x[a])
