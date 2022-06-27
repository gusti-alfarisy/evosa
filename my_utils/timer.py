import time

class Timer:
    def __init__(self, max_time=1000, ttype="iters"):
        self.current_time = 0
        self.max_time = max_time
        self.type = ttype
        if ttype == "iters":
            self.next = self.next_iter
        elif ttype == "seconds":
            self.next = self.next_seconds
            self.last_time = time.time()
        elif ttype == "minutes":
            self.max_time = max_time * 60
            self.next = self.next_seconds
            self.last_time = time.time()

    def reset(self):
        self.current_time = 0

    def start(self):
        self.reset()

    def next_iter(self):
        self.current_time += 1

    def next_seconds(self):
        self.current_time = time.time() - self.last_time

    def end(self):
        return self.current_time >= self.max_time

    def run(self):
        return self.current_time < self.max_time