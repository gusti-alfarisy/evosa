import numpy as np

from encoder.tapotl import TLEncoding
from my_utils.logger import Logger
from my_utils.timer import Timer
import pickle


class Solution2:

    def __init__(self, p,
                 cost=None,
                 cost_train=None,
                 cost_val=None,
                 cost_test=None,
                 cost_top5_train=None,
                 cost_top5_val=None,
                 cost_top5_test=None,
                 key_str=None,
                 total_params=10,
                 latency=-1):
        self.p = p
        self.cost = cost
        self.cost_train = cost_train
        self.cost_val = cost_val
        self.cost_test = cost_test
        self.key_str = key_str
        self.total_params = total_params
        self.latency = latency
        self.cost_top5_train = cost_top5_train
        self.cost_top5_val = cost_top5_val
        self.cost_top5_test = cost_top5_test

    def copy(self, other):
        self.p = np.copy(other.p)
        self.cost = other.cost
        self.cost_train = other.cost_train
        self.cost_val = other.cost_val
        self.cost_test = other.cost_test
        self.key_str = other.key_str
        self.total_params = other.total_params
        self.latency = other.latency
        self.cost_top5_train = other.cost_top5_train
        self.cost_top5_val = other.cost_top5_val
        self.cost_top5_test = other.cost_top5_test

    def clone(self):
        return Solution2(self.p,
                         self.cost,
                         self.cost_train,
                         self.cost_val,
                         self.cost_test,
                         self.cost_top5_train,
                         self.cost_top5_val,
                         self.cost_top5_test,
                         self.key_str,
                         self.total_params,
                         self.latency)

    def __str__(self):
        # print("cost top5 test", self.cost_top5_test)
        return f"Solution: {self.key_str} | Cost: {self.cost} | Train Acc: {(1 - self.cost_train) * 100} % | Validation Acc: {(1 - self.cost_val) * 100} % | Testing Acc: {(1 - self.cost_test) * 100} % | Top-5 Training Acc: {(1 - self.cost_top5_train) * 100} % | Top-5 Val Acc: {(1 - self.cost_top5_val) * 100} % | Top-5 Test Acc: {(1 - self.cost_top5_test) * 100} % | Total Params: {self.total_params / 1_000_000} M | Latency: {self.latency} s"


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



def update_temp(T_curr, i, max_time):
    return T_curr * (max_time - i) / max_time


# # 0 < -nconstant < 1
# def update_temp_standard(T0, i, nconstat=0.01):
#     return -nconstat * i + T0


# np.log is ln in numpy, whereas np.log10 is base 10 log
def boltzmann_annealing(T0, i):
    return T0 / np.log(1 + i)


def SA(max_time, ttype="iters", Objective=None, T=0.3, Encoder=TLEncoding,
       lb=None, ub=None, log_message=True, log_stored_path="SA_topology.csv",
       best_stored_path="SA.obj"):
    log_stored_path = f"output/logs/{log_stored_path}"

    fc = Encoder()
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
    T_curr = T
    log = Logger()

    optimized_num_layer = fc.optimized_num_layers()
    len_encoding = len(fc.encoding)
    while t.run():

        next_p = get_neighbour(current_solution.p)
        next_p = np.clip(next_p, lb, ub)
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
        T_curr = boltzmann_annealing(T, t.current_time+1)
        note = 'Normal'

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
            note = 'Accept Bad'
            print("Accept bad solution -->")

        if log_message:
            print("current solution: ", current_solution)

        log.add("NOID", t.current_time + 1, current_solution.p, current_solution.cost, current_solution.cost_train,
                current_solution.cost_val, current_solution.total_params, current_solution.latency, note)
        log.save_csv(log_stored_path)
        t.next()

    save_pickle(best_solution, f"output/saved_objects/{best_stored_path}")
    return best_solution
