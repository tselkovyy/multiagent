import random
from dataclasses import dataclass
from typing import List, Dict, Deque
from collections import deque

# =======================
# Стоимости операций
# =======================
C_MSG = 0.1
C_CENTER = 1000.0
C_OP = 0.01
C_MEM = 0.1

# =======================
# Параметры
# =======================
MIN_N = 5
MAX_N = 20
MAX_DEG = 3
REL_EPS = 0.01
MAX_ITERS = 30000


@dataclass
class Packet:
    src: int
    value: float
    step: int


class Meter:
    def __init__(self):
        self.msg = 0
        self.center = 0
        self.ops = 0
        self.mem = 0

    def cost(self) -> float:
        return (
            self.msg * C_MSG +
            self.center * C_CENTER +
            self.ops * C_OP +
            self.mem * C_MEM
        )


class Node:
    def __init__(self, idx: int, val: float):
        self.idx = idx
        self.val = val
        self.links: List[int] = []
        self.box: Deque[Packet] = deque()


def make_graph(n: int) -> Dict[int, List[int]]:
    g = {i: set() for i in range(n)}

    # связность
    for i in range(n - 1):
        g[i].add(i + 1)
        g[i + 1].add(i)

    # случайные рёбра
    for i in range(n):
        while len(g[i]) < MAX_DEG:
            j = random.randrange(n)
            if j != i and len(g[j]) < MAX_DEG:
                g[i].add(j)
                g[j].add(i)
            else:
                break

    return {i: list(v) for i, v in g.items()}


def run_consensus():
    rnd = random.Random()
    meter = Meter()

    n = rnd.randint(MIN_N, MAX_N)
    values = [rnd.uniform(0, 100) for _ in range(n)]
    true_avg = sum(values) / n

    graph = make_graph(n)

    nodes: Dict[int, Node] = {}
    for i in range(n):
        nodes[i] = Node(i, values[i])
        nodes[i].links = graph[i]

    meter.mem += n * 2

    step = 0
    while step < MAX_ITERS:
        # рассылка
        for i, node in nodes.items():
            for j in node.links:
                nodes[j].box.append(Packet(i, node.val, step))
                meter.msg += 1
                meter.mem += 1

        new_vals = {}
        max_err = 0.0

        # обработка
        for i, node in nodes.items():
            s = node.val
            cnt = 1

            while node.box and node.box[0].step == step:
                p = node.box.popleft()
                s += p.value
                cnt += 1
                meter.ops += 1

            meter.ops += 1  # деление
            nv = s / cnt
            new_vals[i] = nv

            err = abs(nv - true_avg)
            max_err = max(max_err, err)

        for i in nodes:
            nodes[i].val = new_vals[i]

        if max_err <= REL_EPS * true_avg:
            break

        step += 1

    meter.center += 1

    final_avg = sum(n.val for n in nodes.values()) / n

    # =======================
    # Вывод
    # =======================
    print("===== CONFIG =====")
    print(f"agents: {n}")
    print(f"iterations: {step}")
    print(f"target relative error: {REL_EPS}")

    print("\n===== INITIAL VALUES =====")
    for i, v in enumerate(values):
        print(f"agent{i}: {v:.3f}")

    print("\n===== TOPOLOGY =====")
    for i in range(n):
        print(f"agent{i}: {graph[i]}")

    print("\n===== RESULT =====")
    print(f"true average:      {true_avg:.6f}")
    print(f"computed average:  {final_avg:.6f}")
    print(f"relative error:    {abs(final_avg - true_avg) / true_avg:.6%}")

    print("\n===== COST =====")
    print(f"messages: {meter.msg} -> {meter.msg * C_MSG:.2f}")
    print(f"center msg: {meter.center} -> {meter.center * C_CENTER:.2f}")
    print(f"arith ops: {meter.ops} -> {meter.ops * C_OP:.2f}")
    print(f"memory: {meter.mem} -> {meter.mem * C_MEM:.2f}")
    print(f"TOTAL COST: {meter.cost():.2f}")


if __name__ == "__main__":
    run_consensus()
