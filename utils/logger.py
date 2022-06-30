import pandas as pd
import copy


class Logger:

    def __init__(self):
        self.temp = {"iteration": [], "id": [], "position": [], "cost": [], "cost_train": [], "cost_val": [],
                     "total_params": [],
                     "latency": [], "note": []}
        self.data = copy.deepcopy(self.temp)

    def __add_at(self, key, value):
        self.data[key].append(value)

    def clear_log(self):
        self.data = copy.deepcopy(self.temp)

    def add(self, id_data, iter, position, cost, cost_train, cost_val, total_params, latency, note):
        self.__add_at("id", id_data)
        self.__add_at("iteration", iter)
        self.__add_at("position", position)
        self.__add_at("cost", cost)
        self.__add_at("note", note)
        self.__add_at("cost_train", cost_train)
        self.__add_at("cost_val", cost_val)
        self.__add_at("total_params", total_params)
        self.__add_at("latency", latency)

    def save_csv(self, path):
        df = pd.DataFrame(self.data)
        df.to_csv(path)

    def __str__(self):
        return f"Log Data: {self.data}"


class LoggerEvoSA:

    def __init__(self, keys=[]):
        if len(keys) <= 0:
            keys = [
                "iteration",
                "id",
                "key_str1",
                "cost1",
                "cost_train1",
                "cost_val1",
                "cost_test1",
                "total_params1",
                "latency1",
                "note1",
                "key_str2",
                "cost2",
                "cost_train2",
                "cost_val2",
                "cost_test2",
                "total_params2",
                "latency2",
                "note2"
            ]
        self.temp = {}
        for e in keys:
            self.temp[e] = []

        self.data = copy.deepcopy(self.temp)

    def __add_at(self, key, value):
        self.data[key].append(value)

    def clear_log(self):
        self.data = copy.deepcopy(self.temp)

    def add(self, id_data, iter, key_str1,
            cost1, cost_train1, cost_val1, cost_test1,
            total_params1, latency1, note1,
            key_str2, cost2, cost_train2, cost_val2, cost_test2,
            total_params2, latency2, note2):

        self.__add_at("id", id_data)
        self.__add_at("iteration", iter)
        self.__add_at("key_str1", key_str1)
        self.__add_at("key_str2", key_str2)
        self.__add_at("cost1", cost1)
        self.__add_at("cost2", cost2)
        self.__add_at("note1", note1)
        self.__add_at("note2", note2)
        self.__add_at("cost_train1", cost_train1)
        self.__add_at("cost_train2", cost_train2)
        self.__add_at("cost_val1", cost_val1)
        self.__add_at("cost_val2", cost_val2)
        self.__add_at("cost_test1", cost_test1)
        self.__add_at("cost_test2", cost_test2)
        self.__add_at("total_params1", total_params1)
        self.__add_at("total_params2", total_params2)
        self.__add_at("latency1", latency1)
        self.__add_at("latency2", latency2)

    def save_csv(self, path):
        df = pd.DataFrame(self.data)
        df.to_csv(path)

    def __str__(self):
        return f"Log Data: {self.data}"


if __name__ == "__main__":
    log = Logger()
    log.add("NOID", 1, [0, 0, 1], 0.5, "note nothing")
    log.add("NOID", 2, [0, 0, 2], 0.9, "note nothing")
    log.save_csv("test.csv")
    print(log)
