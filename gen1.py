import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from deap import base, creator, tools, algorithms
import random

# ===============================
# 1. Crear el laberinto
# ===============================
laberinto = np.zeros((20, 20))
inicio = (0, 0)
salida = (0, 19)

for i in range(1, 19, 4):
    laberinto[0:19, i] = 1
for i in range(3, 20, 4):
    laberinto[1:20, i] = 1

# Movimientos posibles: ↑ ↓ ← →
MOVIMIENTOS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


# ===============================
# 2. Funciones auxiliares
# ===============================
def mover(pos, movimiento):
    """Devuelve nueva posición si es válida."""
    x, y = pos
    dx, dy = movimiento
    nx, ny = x + dx, y + dy
    if 0 <= nx < 20 and 0 <= ny < 20 and laberinto[nx, ny] == 0:
        return (nx, ny)
    return pos  # si choca, se queda


def distancia(p1, p2):
    """Distancia Manhattan."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def evaluar(ind):
    """Evalúa qué tan cerca llega el individuo a la salida."""
    pos = inicio
    for m in ind:
        pos = mover(pos, MOVIMIENTOS[m])
        if pos == salida:
            return (1000,)  # gran recompensa si llega
    return (100 - distancia(pos, salida),)


# ===============================
# 3. Configurar DEAP
# ===============================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_move", random.randint, 0, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_move, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluar)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


# ===============================
# 4. Mostrar la mejor ruta
# ===============================
def mostrar_ruta(individuo):
    pos = inicio
    camino = [pos]
    for movimiento in individuo:
        pos = mover(pos, MOVIMIENTOS[movimiento])
        camino.append(pos)

    plt.imshow(laberinto.T, cmap="binary")
    camino = np.array(camino)
    plt.gca().invert_yaxis()
    plt.plot(camino[:, 0], camino[:, 1], "r-", label="Ruta")
    plt.plot(inicio[0], inicio[1], "bo", label="Inicio")
    plt.plot(salida[0], salida[1], "go", label="Salida")
    plt.legend()
    plt.title("Ruta encontrada")
    plt.show()


# ===============================
# 5. Ejecución con multiprocessing
# ===============================
if __name__ == "__main__":
    # Parámetros ajustables:
    TAM_POB = 200       # población
    NGEN = 1000          # generaciones
    PROB_CX = 0.8         # prob. cruce
    PROB_MUT = 0.8        # prob. mutación

    # Pool de procesos
    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)

        pop = toolbox.population(n=TAM_POB)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        print("Ejecutando algoritmo genético...")
        pop, logbook = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=PROB_CX,
            mutpb=PROB_MUT,
            ngen=NGEN,
            stats=stats,
            verbose=True
        )

    mejor = tools.selBest(pop, 1)[0]
    print("\nMejor fitness:", mejor.fitness.values[0])
    mostrar_ruta(mejor)
