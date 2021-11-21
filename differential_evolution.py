import matplotlib.pyplot as plt
import random
import seaborn as sns

POPULATION_SIZE = 20
LEFT_BOUND = -5
RIGHT_BOUND = 5
t = 1
F = 0.5
K = 0.5
PC = 0.5
GENERATION = 1000

def func(x1, x2):
    return (100 * (x2 - x1**2)**2 + (1-x1)**2)


def current_to_pbest(population, fitness, best_candidate):
    mutant_vec = []
    mutant_fitness = []
    for i in range(POPULATION_SIZE):
        r1 = i
        r2 = random.randrange(POPULATION_SIZE)
        while(r2==r1):
            r2 = random.randrange(POPULATION_SIZE)
        r3 = random.randrange(POPULATION_SIZE)
        while(r3==r1 or r3==r2):
            r3 = random.randrange(POPULATION_SIZE)

        mutant = [population[r1][0] + F*(best_candidate[0] - population[r1][0] + population[r2][0]-population[r3][0]), population[r1][1] + F*(best_candidate[1] - population[r1][1] + population[r2][1]-population[r3][1])]
        if mutant[0]>RIGHT_BOUND:
            mutant[0]=RIGHT_BOUND
        if mutant[0]<LEFT_BOUND:
            mutant[0]=LEFT_BOUND
        if mutant[1]>RIGHT_BOUND:
            mutant[1]=RIGHT_BOUND
        if mutant[1]<LEFT_BOUND:
            mutant[1]=LEFT_BOUND

        mutant_vec.append((mutant[0], mutant[1]))
        mutant_fitness.append(func(mutant[0], mutant[1]))
    


def current_to_rand(population, fitness):
    mutant_vec = []
    mutant_fitness = []
    for i in range(POPULATION_SIZE):
        r1 = i
        r2 = random.randrange(POPULATION_SIZE)
        while(r2==r1):
            r2 = random.randrange(POPULATION_SIZE)
        r3 = random.randrange(POPULATION_SIZE)
        while(r3==r1 or r3==r2):
            r3 = random.randrange(POPULATION_SIZE)
        r4 = random.randrange(POPULATION_SIZE)
        while(r4==r1 or r4==r2 or r4==r3):
            r4 = random.randrange(POPULATION_SIZE)

        mutant = [population[r1][0] + K*(population[r4][0] - population[r1][0]) + F*(population[r2][0]-population[r3][0]), population[r1][1] + K*(population[r4][1] - population[r1][1]) + F*(population[r2][1]-population[r3][1])]
        if mutant[0]>RIGHT_BOUND:
            mutant[0]=RIGHT_BOUND
        if mutant[0]<LEFT_BOUND:
            mutant[0]=LEFT_BOUND
        if mutant[1]>RIGHT_BOUND:
            mutant[1]=RIGHT_BOUND
        if mutant[1]<LEFT_BOUND:
            mutant[1]=LEFT_BOUND

        mutant_vec.append((mutant[0], mutant[1]))
        mutant_fitness.append(func(mutant[0], mutant[1]))

def rand(population, fitness):
    mutant_vec = []
    mutant_fitness = []
    for i in range(POPULATION_SIZE):
        r1 = i
        r2 = random.randrange(POPULATION_SIZE)
        while(r2==r1):
            r2 = random.randrange(POPULATION_SIZE)
        r3 = random.randrange(POPULATION_SIZE)
        while(r3==r1 or r3==r2):
            r3 = random.randrange(POPULATION_SIZE)

        mutant = [population[r1][0] + F*(population[r2][0]-population[r3][0]), population[r1][1] + F*(population[r2][1]-population[r3][1])]
        if mutant[0]>RIGHT_BOUND:
            mutant[0]=RIGHT_BOUND
        if mutant[0]<LEFT_BOUND:
            mutant[0]=LEFT_BOUND
        if mutant[1]>RIGHT_BOUND:
            mutant[1]=RIGHT_BOUND
        if mutant[1]<LEFT_BOUND:
            mutant[1]=LEFT_BOUND

        mutant_vec.append((mutant[0], mutant[1]))
        mutant_fitness.append(func(mutant[0], mutant[1]))


def mutate(population, fitness):
    mutant_vec = []
    mutant_fitness = []
    for i in range(POPULATION_SIZE):
        r1 = i
        r2 = random.randrange(POPULATION_SIZE)
        while(r2==r1):
            r2 = random.randrange(POPULATION_SIZE)
        r3 = random.randrange(POPULATION_SIZE)
        while(r3==r1 or r3==r2):
            r3 = random.randrange(POPULATION_SIZE)

        mutant = [population[r1][0] + F*(population[r2][0]-population[r3][0]), population[r1][1] + F*(population[r2][1]-population[r3][1])]
        if mutant[0]>RIGHT_BOUND:
            mutant[0]=RIGHT_BOUND
        if mutant[0]<LEFT_BOUND:
            mutant[0]=LEFT_BOUND
        if mutant[1]>RIGHT_BOUND:
            mutant[1]=RIGHT_BOUND
        if mutant[1]<LEFT_BOUND:
            mutant[1]=LEFT_BOUND

        mutant_vec.append((mutant[0], mutant[1]))
        mutant_fitness.append(func(mutant[0], mutant[1]))

    new_pop = []
    new_pop_fitness = []
    for i in range(POPULATION_SIZE):
        temp = (random.random(), random.random())
        x = [0, 0]
        if temp[0]>PC:
            x[0] = mutant_vec[i][0]
        else:
            x[0] = population[i][0]
        
        if temp[1]>PC:
            x[1] = mutant_vec[i][1]
        else:
            x[1] = population[i][1]
        
        new_pop.append((x[0], x[1]))
        new_pop_fitness.append(func(x[0], x[1]))

    for i in range(0, POPULATION_SIZE):
        if (abs(fitness[i]) > abs(new_pop_fitness[i])):
            population[i] = new_pop[i]
            fitness[i] = new_pop_fitness[i]

    

population = []
fitness = []

for i in range(POPULATION_SIZE):
    temp = (random.uniform(LEFT_BOUND, RIGHT_BOUND), random.uniform(LEFT_BOUND, RIGHT_BOUND))
    print(temp)
    population.append((random.uniform(LEFT_BOUND, RIGHT_BOUND), random.uniform(LEFT_BOUND, RIGHT_BOUND)))

    fitness.append(round(func(temp[0], temp[1]), 3))


for g in range(GENERATION):
    old_pop = population.copy()
    old_fitness = fitness.copy()
    mutate(population, fitness)

    print('Generation:', g+1)
    # print('population')

    # for i in range(POPULATION_SIZE):
    #     print(population[i], old_pop[i])

    # print('fitness')
    best_fitness = fitness[0]
    x=-1
    for i in range(POPULATION_SIZE):
        #print(fitness[i], old_fitness[i])
        if abs(fitness[i]) < abs(best_fitness):
            best_fitness = fitness[i]
            x = i
    # print('element:', population[x],'fitness', best_fitness)
    sns.scatterplot(x=[population[x][0]], y=[population[x][1]], color='green', s=150)
    x = [population[i][0] for i in range(POPULATION_SIZE)]
    y = [population[i][1] for i in range(POPULATION_SIZE)]
    # for i in range(POPULATION_SIZE):
    #     print('(', population[i][0], ',', population[i][1], ')', fitness[i])

    sns.scatterplot(x=x, y=y)
    sns.scatterplot(x=[1], y=[1], color='red')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

    

print('Ideal solution:\nelements: (1, 1)')
print('fitness: ', func(1, 1))
x = [population[i][0] for i in range(POPULATION_SIZE)]
y = [population[i][1] for i in range(POPULATION_SIZE)]
sns.scatterplot(x=x, y=y)
sns.scatterplot(x=[1], y=[1], color='red')
plt.show()
