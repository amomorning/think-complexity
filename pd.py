import random
import numpy as np



class Agent:
    # C - cooperate; D - defect
    # 'CCCCCCC' for those always cooperate
    # 'CCDCDCD' for TFT
    # 'CCCCCCD' for forgive TFT
    # (agent, opponent)
    keys = [(None, None), (None, 'C'), (None, 'D'), 
        ('C', 'C'), ('C', 'D'), ('D', 'C'), ('D', 'D')]
    def __init__(self, values, fitness = np.nan):
        self.values = values
        self.response = dict(zip(self.keys, values))
        self.fitness = fitness
    
    def reset(self):
        self.hist = [None, None]
        self.score = 0
    
    def respond(self, other):
        return self.response[tuple(other.hist[-2:])]
    
    def add(self, response, score):
        self.hist.append(response)
        self.score += score

    def copy(self, prob_mutate=0.05):
        if random.random() > prob_mutate:
            values = self.values
        else:
            values = self.mutate()
        return Agent(values, self.fitness)
    
    def mutate(self):
        values = list(self.values)
        idx = np.random.choice(len(values))
        values[idx] = 'C' if values[idx] == 'D' else 'D'
        return values


payoffs = {('C', 'C'): (3, 3), 
           ('C', 'D'): (0, 5), 
           ('D', 'C'): (5, 0), 
           ('D', 'D'): (1, 1)}
        

num_rounds = 6
def play(a, b):
    a.reset()
    b.reset()

    for _ in range(num_rounds):
        ra = a.respond(b)
        rb = b.respond(a)
        
        sa, sb = payoffs[ra, rb]
        a.add(ra, sa)
        b.add(rb, sb)

    return a.score, b.score


def melee(agents):
    n = len(agents)
    tot = np.zeros(n)
    for i in range(n):
        a, b = agents[i], agents[(i+1)%n]
        sa, sb = play(a, b)
        tot[i] += sa
        tot[(i+1)%n] += sb
    for i in range(n):
        agents[i].fitness = tot[i] / num_rounds / 2
        
def logistic(x, a=0.7, b=1.5, m=2.5, k=0.9):
    exp = -b * (x - m)
    denom = 1 + np.exp(exp)
    return a + (k-a) / denom


def run(n=100, m=500):
    agents = [Agent(np.random.choice(['C', 'D'], size=7)) 
              for _ in range(n)]
    def step():
        melee(agents)
        fits = np.array([agent.fitness for agent in agents])
        ps = logistic(fits)
        is_dead = np.random.random(n) < ps
        idx = np.nonzero(is_dead)[0]
        rest = np.nonzero(np.logical_not(is_dead))[0]

        for p in idx:
            q = np.random.choice(rest, 1)[0]
            agents[p] = agents[q].copy()
    
    for i in range(m):
        step()
    
    mp = {}
    for i in range(n):
        value = ''.join(agents[i].values)
        mp[value] = mp.get(value, 0) + 1
    print(mp)
        
run()

