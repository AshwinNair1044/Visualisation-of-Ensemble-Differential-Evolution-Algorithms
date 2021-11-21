import matplotlib.pyplot as plt
import random
import seaborn as sns
import math
import numpy as np
import pandas as pd

POPULATION_SIZE = 20
LEFT_BOUND = -5
RIGHT_BOUND = 5
t = 1
F = 0.5
K = 0.5
PC = 0.5
GENERATION = 50


def Schwefel(x1, x2): 
    return -1 * (x1 * math.sin(math.sqrt(abs(x1))) + (x2 * math.sin(math.sqrt(abs(x2)))))

def Michalewicz(x1, x2):
    return -1 * (math.sin(x1) * math.sin((1 * x1**2)/180)**20) * (math.sin(x2) * math.sin((1 * x2**2)/180)**20)


def quadric(x1, x2):
    return x1**2 + (x1 + x2)**2

def salomon(x1, x2):
    temp = x1**2 + x2**2
    temp = math.sqrt(temp)
    f = (-1*math.cos(np.pi*temp)) + (0.1 * temp) + 1
    return f

def Rana(x1, x2):
    alpha = math.sqrt(abs(x2 + 1 -x1))
    beta = math.sqrt(abs(x1 + x2 + 1))
    f = (x1 * math.sin(alpha) * math.cos(beta)) + ((x1+1) * math.cos(alpha) * math.sin(beta))
    return f



def func(x1, x2):
    return (100 * (x2 - x1**2)**2 + (1-x1)**2)


def current_to_pbest(population, fitness, best_candidate=(0, 0)):
    mutant_vec = []
    mutant_fitness = []
    for i in range(POPULATION_SIZE//4):
        r1 = i
        r2 = random.randrange(POPULATION_SIZE//4)
        while(r2==r1):
            r2 = random.randrange(POPULATION_SIZE//4)
        r3 = random.randrange(POPULATION_SIZE//4)
        while(r3==r1 or r3==r2):
            r3 = random.randrange(POPULATION_SIZE//4)

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
    
    new_pop = []
    new_pop_fitness = []
    for i in range(POPULATION_SIZE//4):
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

    for i in range(0, POPULATION_SIZE//4):
        if (abs(fitness[i]) > abs(new_pop_fitness[i])):
            population[i] = new_pop[i]
            fitness[i] = new_pop_fitness[i]


def current_to_rand(population, fitness):
    mutant_vec = []
    mutant_fitness = []
    for i in range(POPULATION_SIZE//4):
        r1 = i
        r2 = random.randrange(POPULATION_SIZE//4)
        while(r2==r1):
            r2 = random.randrange(POPULATION_SIZE//4)
        r3 = random.randrange(POPULATION_SIZE//4)
        while(r3==r1 or r3==r2):
            r3 = random.randrange(POPULATION_SIZE//4)
        r4 = random.randrange(POPULATION_SIZE//4)
        while(r4==r1 or r4==r2 or r4==r3):
            r4 = random.randrange(POPULATION_SIZE//4)

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

    new_pop = []
    new_pop_fitness = []
    for i in range(POPULATION_SIZE//4):
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

    for i in range(0, POPULATION_SIZE//4):
        if (abs(fitness[i]) > abs(new_pop_fitness[i])):
            population[i] = new_pop[i]
            fitness[i] = new_pop_fitness[i]

def rand(population, fitness):
    mutant_vec = []
    mutant_fitness = []
    for i in range(POPULATION_SIZE//4):
        r1 = i
        r2 = random.randrange(POPULATION_SIZE//4)
        while(r2==r1):
            r2 = random.randrange(POPULATION_SIZE//4)
        r3 = random.randrange(POPULATION_SIZE//4)
        while(r3==r1 or r3==r2):
            r3 = random.randrange(POPULATION_SIZE//4)

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
    for i in range(POPULATION_SIZE//4):
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

    for i in range(0, POPULATION_SIZE//4):
        if (abs(fitness[i]) > abs(new_pop_fitness[i])):
            population[i] = new_pop[i]
            fitness[i] = new_pop_fitness[i]


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
    population.append((random.uniform(LEFT_BOUND, RIGHT_BOUND), random.uniform(LEFT_BOUND, RIGHT_BOUND)))

    fitness.append(round(func(temp[0], temp[1]), 3))


pop1_s = pop2_s = pop3_s = POPULATION_SIZE//4
pop4_s = POPULATION_SIZE - pop1_s - pop2_s - pop3_s


pop1 = population[0:pop1_s]
pop2 = population[pop1_s:2*pop1_s]
pop3 = population[2*pop1_s:3*pop1_s]
pop4 = population[3*pop1_s:]

# print(len(pop1))
# print(len(pop2))
# print(len(pop3))
# print(len(pop4))

gene = []
x_temp = []
y_temp = []
fit_temp = []


best_population = (0.0, 0.0)

for g in range(GENERATION):
    # old_pop = population.copy()
    # old_fitness = fitness.copy()
    # mutate(population, fitness)
    old_pop1 = pop1.copy()
    fit1 = []
    fit2 = []
    fit3 = []
    for i in pop1:
        fit1.append(func(i[0], i[1]))
    for i in pop2:
        fit2.append(func(i[0], i[1]))
    for i in pop3:
        fit3.append(func(i[0], i[1]))
    old_pop1 = pop1.copy()
    old_pop1 = pop1.copy()
    old_pop1 = pop1.copy()
    current_to_pbest(pop1, fit1, best_population)
    current_to_rand(pop2, fit2)
    rand(pop3, fit3)

    
        
    population = pop1 + pop2+pop3+pop4
    fitness = []
    for i in population:
        fitness.append(func(i[0], i[1]))
    print('Generation:', g+1)
    # print('population')

    # for i in range(POPULATION_SIZE):
    #     print(population[i], old_pop[i])

    # print('fitness')
    best_fitness = fitness[0]
    x=0
    for i in range(POPULATION_SIZE):
        #print(fitness[i], old_fitness[i])
        if abs(func(population[i][0], population[i][1])) < abs(best_fitness):
            best_fitness = func(population[i][0], population[i][1])
            x = i
    # print('element:', population[x],'fitness', best_fitness)
    sns.scatterplot(x=[population[x][0]], y=[population[x][1]], color='green', s=150)
    x = [population[i][0] for i in range(POPULATION_SIZE)]
    y = [population[i][1] for i in range(POPULATION_SIZE)]
    # for i in range(POPULATION_SIZE):
    #     print('(', population[i][0], ',', population[i][1], ')', fitness[i])

    best_fitness1 = fit1[0]
    b = 0
    x1=-1
    for i in range(POPULATION_SIZE//4):
        #print(fitness[i], old_fitness[i])
        if abs(fit1[i]) < abs(best_fitness1):
            best_fitness1 = fit1[i]
            x1 = i
    best_population = pop1[x1]

    if (g%5==0):
        best_fitness1 = fit1[0]
        b = 0
        x1=-1
        for i in range(POPULATION_SIZE//4):
            #print(fitness[i], old_fitness[i])
            if abs(fit1[i]) < abs(best_fitness1):
                best_fitness1 = fit1[i]
                x1 = i
        best_fitness2 = fit2[0]
        x2=-1
        for i in range(POPULATION_SIZE//4):
            #print(fitness[i], old_fitness[i])
            if abs(fit2[i]) < abs(best_fitness2):
                best_fitness2 = fit2[i]
                x1 = i
        best_fitness3 = fit3[0]
        x3=-1
        for i in range(POPULATION_SIZE//4):
            #print(fitness[i], old_fitness[i])
            if abs(fit3[i]) < abs(best_fitness3):
                best_fitness3 = fit3[i]
                x1 = i
        if best_fitness1 > best_fitness2:
            if best_fitness1>best_fitness3:
                b = 1
            else:
                b = 3
        else:
            if best_fitness2>best_fitness3:
                b = 2
            else:
                b = 3
        if b==1:
            pop1 = pop1 + pop4
            random.shuffle(pop1)
            pop4 = pop1[:len(pop1)//2]
            pop1 = pop1[len(pop1)//2:]
        if b==2:
            pop2 = pop2 + pop4
            random.shuffle(pop2)
            pop4 = pop2[:len(pop2)//2]
            pop2 = pop2[len(pop2)//2:]
        if b==3:
            pop3 = pop3 + pop4
            random.shuffle(pop3)
            pop4 = pop3[:len(pop3)//2]
            pop3 = pop3[len(pop3)//2:]
    u3 = []
    u = []
    u1 = []
    u2 = []
    for h in range(POPULATION_SIZE):
        u3.append(g)
        u.append(population[h][0])
        u1.append(population[h][1])
        u2.append(func(population[h][0], population[h][1]))

    gene = gene + u3
    x_temp = x_temp + u
    y_temp = y_temp + u1
    fit_temp = fit_temp + u2
    
    sns.scatterplot(x=x, y=y)
    sns.scatterplot(x=[1], y=[1], color='red')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

    
print(len(gene))
print(len(x_temp))
print(len(y_temp))
print(len(fit_temp))
df = pd.DataFrame({'Generation': gene, 'X': x_temp, 'Y': y_temp, 'Fitness': fit_temp})
df.to_csv(r'custom_quadric.csv', index = False, header = True)
print('csv saved')

print('Ideal solution:\nelements: (1, 1)')
print('fitness: ', func(1, 1))
x = [population[i][0] for i in range(POPULATION_SIZE)]
y = [population[i][1] for i in range(POPULATION_SIZE)]
sns.scatterplot(x=x, y=y)
sns.scatterplot(x=[1], y=[1], color='red')
plt.show()
