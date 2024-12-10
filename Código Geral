import bpy
import os
import numpy as np
import math

# Função para carregar sólidos de um arquivo
def carregar_solidos(arquivo):
    solidos = {}
    with open(arquivo, 'r') as f:
        linhas = f.readlines()
        for linha in linhas:
            if " | n=" in linha:
                partes = linha.split(" | n=")
                nome_e_faces = partes[0].strip()
                n = int(partes[1].strip())
                
                nome, faces_raw = nome_e_faces.split(" - ")
                faces = eval(faces_raw)  # Use com cuidado, ou substitua por `json.loads` para maior segurança
                
                solidos[nome] = (faces, n)
    return solidos

# Função para selecionar um sólido pelo nome
def selecionar_solido(solidos, nome):
    if nome in solidos:
        return solidos[nome]
    else:
        raise ValueError(f"Sólido '{nome}' não encontrado no arquivo.")

# Função para atualizar o objeto no Blender
def atualizar_objeto(vertices, edges, faces):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    info_objeto = bpy.data.meshes.new('info')
    objeto = bpy.data.objects.new('sólido', info_objeto)
    bpy.context.scene.collection.objects.link(objeto)
    bpy.context.view_layer.objects.active = objeto
    bpy.context.active_object.select_set(state=True)
    info_objeto.from_pydata(vertices, edges, faces)
    info_objeto.update()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

# Funções auxiliares para o cálculo
def d2(vc, i, j):
    return np.sum((vc[i] - vc[j]) ** 2)

def calcular_d2(n, i, j):
    if n == 3:
        return 1
    elif n == 4:
        return math.sqrt(2) ** 2 if abs(i - j) == 2 else 1
    elif n == 5:
        return ((1 + math.sqrt(5)) / 2) ** 2 if abs(i - j) in [2, 3] else 1
    elif n == 6:
        return (math.sqrt(3)) ** 2 if abs(i - j) == 3 else (4 if abs(i - j) == 2 else 1)
    elif n == 8:
        if abs(i - j) == 4:
            return (1 / math.sqrt(1 / 2 - math.sqrt(2) / 4)) ** 2
        elif abs(i - j) == 3:
            return (math.sqrt(math.sqrt(2) / 4 + 1 / 2) / math.sqrt(1 / 2 - math.sqrt(2) / 4)) ** 2
        elif abs(i - j) == 2:
            return (math.sqrt(2) / (2 * math.sqrt(1 / 2 - math.sqrt(2) / 4))) ** 2
        else:
            return 1
    elif n == 10:
        if abs(i - j) == 5:
            return (2 / (-1 / 2 + math.sqrt(5) / 2)) ** 2
        elif abs(i - j) == 4:
            return (2 * math.sqrt(math.sqrt(5) / 8 + 5 / 8) / (-1 / 2 + math.sqrt(5) / 2)) ** 2
        elif abs(i - j) == 3:
            return (2 * (1 / 4 + math.sqrt(5) / 4) / (-1 / 2 + math.sqrt(5) / 2)) ** 2
        elif abs(i - j) == 2:
            return (2 * math.sqrt(5 / 8 - math.sqrt(5) / 8) / (-1 / 2 + math.sqrt(5) / 2)) ** 2
        else:
            return 1

def funcao(vc, segmentos):
    soma = 0
    for p in segmentos:
        i, j, d2_valor = p
        soma += (d2(vc, i, j) - d2_valor) ** 2
    return soma

def calcular_incidencias(num_de_vertices, segmentos):
    incidencias = [[] for _ in range(num_de_vertices)]
    for idx, (i, j, _) in enumerate(segmentos):
        incidencias[i].append(idx)
        incidencias[j].append(idx)
    return incidencias

def gradiente(vc, segmentos, incidencias):
    n = len(vc)
    grad = np.zeros_like(vc)
    for idx, (i, j, d2_valor) in enumerate(segmentos):
        dist = np.sqrt(d2(vc, i, j))
        if dist == 0:  # Para evitar divisão por zero
            continue
        grad[i] += 2 * (dist - math.sqrt(d2_valor)) * (vc[i] - vc[j]) / dist
        grad[j] += 2 * (dist - math.sqrt(d2_valor)) * (vc[j] - vc[i]) / dist
    return grad

# Carregar sólidos do arquivo faces.txt
caminho_arquivo = os.path.join(os.path.dirname(bpy.data.filepath), "faces.txt")
solidos = carregar_solidos(caminho_arquivo)

# Escolher o sólido pelo nome
nome_solido = "J2"  # Altere conforme necessário
faces, n = selecionar_solido(solidos, nome_solido)

# Inicializar vértices e edges
edges = []
num_de_vertices = max(max(f) for f in faces) + 1
vc = np.random.uniform(-1, 1, (num_de_vertices, 3))

# Gerar segmentos
segmentos = []
for f in faces:
    tamanho = len(f)
    for i in range(tamanho):
        for j in range(i + 1, tamanho):
            d2_valor = calcular_d2(tamanho, f[i], f[j])
            segmentos.append([f[i], f[j], d2_valor])

# Otimização
alpha = 0.01
tolerance = 1e-5
max_iterations = 10000
velocity = np.zeros_like(vc)

incidencias = calcular_incidencias(num_de_vertices, segmentos)
norma_grad = 1.0
i = 0

while norma_grad > tolerance and i < max_iterations:
    i += 1
    grad_atual = gradiente(vc, segmentos, incidencias)
    norma_grad = np.linalg.norm(grad_atual)

    velocity -= alpha * grad_atual
    velocity *= 0.99  # Atrito
    vc += velocity

    atualizar_objeto(vc, edges, faces)

print("Otimização concluída.")
