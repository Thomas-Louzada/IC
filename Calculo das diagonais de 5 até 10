import sympy as sp

def raio_circunscrito(num_lados, lado):
    # Calcula o ângulo central do polígono
    angulo_central = sp.pi / num_lados
    
    # Calcula o raio da circunferência circunscrita
    raio = lado / (2 * sp.sin(angulo_central))
    
    return raio

def todas_distancias_entre_vertices(num_lados, raio):
    # Define simbolicamente o número de lados e o raio da circunferência
    n = sp.Symbol('n', integer=True)
    r = sp.Symbol('r')

    # Lista para armazenar as distâncias
    distancias = []

    # Calcula as distâncias entre todos os pares de vértices
    for k in range(1, num_lados // 2 + 1):
        distancia = 2 * r * sp.sin(k * sp.pi / n)
        distancia = distancia.subs({n: num_lados, r: raio})
        distancias.append(distancia)

    return distancias

# Exemplo de uso: Calculando o raio e as distâncias para um hexágono (6 lados) com lado de comprimento 1
num_lados = 10
lado = 1

# Calcula o raio da circunferência circunscrita
raio = raio_circunscrito(num_lados, lado)
print(f"O raio da circunferência circunscrita ao polígono de {num_lados} lados com lado de comprimento {lado} é: {raio}")

# Calcula todas as distâncias entre os vértices
distancias = todas_distancias_entre_vertices(num_lados, raio)
print(f"\nTodas as distâncias entre vértices de um polígono de {num_lados} lados com raio {raio} são:")
for i, distancia in enumerate(distancias, 1):
    print(f"Distância entre vértices separados por {i} lados: {distancia}")
