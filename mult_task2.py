import random
from dataclasses import dataclass
from typing import Dict, List, Deque, Tuple
from collections import deque, defaultdict

# =======================
# Стоимости
# =======================
C_MSG = 0.1
C_CENTER = 1000.0
C_OP = 0.01
C_MEM = 0.1

# =======================
# Параметры
# =======================
MIN_AGENTS = 5
MAX_AGENTS = 20
MAX_DEGREE = 3

REL_EPS = 0.01
MAX_ITERS = 20000

P_LOSS = 0.10
OFFLINE_PROB = 0.05
OFFLINE_DUR = (1, 2)
DELAY_RANGE = (1, 2)

ALPHA = 0.25


# =======================
# Сообщение
# =======================
@dataclass
class Msg:
    src: int
    dst: int
    value: float
    deliver_at: int


# =======================
# Счётчик стоимости
# =======================
class Cost:
    def __init__(self):
        self.msg = 0
        self.msg_lost = 0
        self.msg_delayed = 0
        self.center = 0
        self.ops = 0
        self.mem = 0

    def total(self) -> float:
        return (
            self.msg * C_MSG +
            self.center * C_CENTER +
            self.ops * C_OP +
            self.mem * C_MEM
        )


# =======================
# Агент
# =======================
class Agent:
    def __init__(self, idx: int, value: float):
        self.idx = idx
        self.x = value
        self.neighbors: List[int] = []
        self.inbox: Deque[Msg] = deque()
        self.offline_until = -1

    def is_offline(self, t: int) -> bool:
        return t <= self.offline_until

    def maybe_fail(self, t: int):
        if self.is_offline(t):
            return
        if random.random() < OFFLINE_PROB:
            self.offline_until = t + random.choice(OFFLINE_DUR)

    def update(self, t: int, cost: Cost) -> float:
        if self.is_offline(t):
            self.inbox.clear()
            return self.x

        vals = []
        while self.inbox and self.inbox[0].deliver_at <= t:
            m = self.inbox.popleft()
            vals.append(m.value)

        if not vals:
            return self.x

        s = 0.0
        for v in vals:
            s += (v - self.x)
        cost.ops += 2 * len(vals)

        dx = ALPHA * s
        cost.ops += 1
        nx = self.x + dx
        cost.ops += 1
        return nx


# =======================
# Граф
# =======================
def build_graph(n: int) -> Dict[int, List[int]]:
    g = {i: set() for i in range(n)}

    for i in range(n - 1):
        g[i].add(i + 1)
        g[i + 1].add(i)

    for i in range(n):
        while len(g[i]) < MAX_DEGREE:
            j = random.randrange(n)
            if j != i and len(g[j]) < MAX_DEGREE:
                g[i].add(j)
                g[j].add(i)
            else:
                break

    return {i: list(v) for i, v in g.items()}


# =======================
# Сеть с помехами
# =======================
class Network:
    def __init__(self):
        self.schedule: Dict[int, List[Msg]] = defaultdict(list)

    def send(self, src: int, dst: int, val: float, t: int, cost: Cost):
        cost.msg += 1

        if random.random() < P_LOSS:
            cost.msg_lost += 1
            return

        delay = random.choice(DELAY_RANGE)
        if delay > 0:
            cost.msg_delayed += 1

        deliver = t + delay
        self.schedule[deliver].append(Msg(src, dst, val, deliver))
        cost.mem += 1

    def deliver(self, t: int) -> Dict[int, List[Msg]]:
        msgs = self.schedule.pop(t, [])
        res = defaultdict(list)
        for m in msgs:
            res[m.dst].append(m)
        return res


# =======================
# Основной алгоритм
# =======================
def run_task2(seed: int = 42):
    random.seed(seed)
    cost = Cost()

    n = random.randint(MIN_AGENTS, MAX_AGENTS)
    init_vals = [random.uniform(0, 100) for _ in range(n)]
    true_mean = sum(init_vals) / n

    graph = build_graph(n)

    agents = {i: Agent(i, init_vals[i]) for i in range(n)}
    for i in range(n):
        agents[i].neighbors = graph[i]

    cost.mem += n * 2
    net = Network()

    t = 0
    while t < MAX_ITERS:
        for a in agents.values():
            a.maybe_fail(t)

        for i, a in agents.items():
            if a.is_offline(t):
                continue
            for nb in a.neighbors:
                net.send(i, nb, a.x, t, cost)

        delivered = net.deliver(t)
        for dst, msgs in delivered.items():
            agents[dst].inbox.extend(msgs)

        new_vals = {}
        max_err = 0.0
        for i, a in agents.items():
            nx = a.update(t, cost)
            new_vals[i] = nx
            max_err = max(max_err, abs(nx - true_mean))

        for i in agents:
            agents[i].x = new_vals[i]

        if max_err <= REL_EPS * true_mean:
            break

        t += 1

    cost.center += 1

    final_vals = [a.x for a in agents.values()]
    est = sum(final_vals) / n

    # =======================
    # Вывод
    # =======================
    print("===== TASK 2: IMPAIRED CONSENSUS =====")
    print(f"agents: {n}")
    print(f"iterations: {t}")
    print(f"loss prob: {P_LOSS}, delays: {DELAY_RANGE}, offline: {OFFLINE_DUR}")
    print("\nInitial values:")
    for i, v in enumerate(init_vals):
        print(f" agent{i}: {v:.2f}")

    print("\nTopology:")
    for i in range(n):
        print(f" agent{i}: {graph[i]}")

    print("\n===== RESULT =====")
    print(f"true mean:      {true_mean:.6f}")
    print(f"estimated mean: {est:.6f}")
    print(f"relative error: {abs(est - true_mean) / true_mean:.2%}")

    print("\n===== COST =====")
    print(f"messages sent: {cost.msg}")
    print(f"messages lost: {cost.msg_lost}")
    print(f"messages delayed: {cost.msg_delayed}")
    print(f"arith ops: {cost.ops}")
    print(f"memory cells: {cost.mem}")
    print(f"TOTAL COST: {cost.total():.2f}")


if __name__ == "__main__":
    run_task2(seed=random.randint(0, 10**9))
