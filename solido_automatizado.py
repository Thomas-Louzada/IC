bl_info = {
    "name": "Gerador de Sólidos Automatizado",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import os
import numpy as np
import math
from collections import defaultdict, deque

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
                faces = eval(faces_raw)
                solidos[nome] = (faces, n)
    return solidos

def corrigir_orientacao_faces(faces_originais):
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces_originais):
        for i in range(len(face)):
            j = (i + 1) % len(face)
            key = tuple(sorted((face[i], face[j])))
            edge_to_faces[key].append(fi)

    oriented = [None] * len(faces_originais)
    processed = [False] * len(faces_originais)
    directed_edges = {}
    queue = deque()

    # Inicialização com a primeira face
    if faces_originais:
        oriented[0] = list(faces_originais[0])
        processed[0] = True
        for i in range(len(oriented[0])):
            j = (i + 1) % len(oriented[0])
            a, b = oriented[0][i], oriented[0][j]
            directed_edges[tuple(sorted((a, b)))] = (a, b)
        queue.append(0)

    # Propagação BFS
    while queue:
        fi = queue.popleft()
        face_ref = oriented[fi]
        for i in range(len(face_ref)):
            j = (i + 1) % len(face_ref)
            a_ref, b_ref = face_ref[i], face_ref[j]
            key = tuple(sorted((a_ref, b_ref)))
            for nbr in edge_to_faces[key]:
                if not processed[nbr] and nbr != fi:
                    face_nbr = list(faces_originais[nbr])
                    # Encontra orientação correta
                    for k in range(len(face_nbr)):
                        x, y = face_nbr[k], face_nbr[(k+1)%len(face_nbr)]
                        if {x, y} == {a_ref, b_ref}:
                            if (x, y) == (b_ref, a_ref):
                                face_nbr = list(reversed(face_nbr))
                            break
                    # Atualiza estruturas
                    oriented[nbr] = face_nbr
                    processed[nbr] = True
                    for m in range(len(face_nbr)):
                        u = face_nbr[m]
                        v = face_nbr[(m+1)%len(face_nbr)]
                        directed_edges[tuple(sorted((u, v)))] = (u, v)
                    queue.append(nbr)
    
    # Fallback para faces não alcançadas
    for i in range(len(oriented)):
        if oriented[i] is None:
            oriented[i] = list(faces_originais[i])
    
    return oriented

def atualizar_objeto(vc, edges, faces):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    mesh = bpy.data.meshes.new("sólido")
    obj = bpy.data.objects.new("sólido", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mesh.from_pydata(vc.tolist(), edges, faces)
    mesh.update()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    return obj

def d2(vc, i, j):
    return np.sum((vc[i] - vc[j]) ** 2)

def calcular_d2(n, i, j):
    if n == 3:
        return 1
    elif n == 4:
        return (math.sqrt(2) ** 2) if abs(i - j) == 2 else 1
    elif n == 5:
        return (((1 + math.sqrt(5)) / 2) ** 2) if abs(i - j) in [2, 3] else 1
    elif n == 6:
        return (math.sqrt(3) ** 2) if abs(i - j) == 3 else (4 if abs(i - j) == 2 else 1)
    elif n == 8:
        if abs(i - j) == 4:
            return (1 / math.sqrt(1/2 - math.sqrt(2)/4)) ** 2
        elif abs(i - j) == 3:
            return (math.sqrt(math.sqrt(2)/4 + 1/2) / math.sqrt(1/2 - math.sqrt(2)/4)) ** 2
        elif abs(i - j) == 2:
            return (math.sqrt(2) / (2 * math.sqrt(1/2 - math.sqrt(2)/4))) ** 2
        else:
            return 1
    elif n == 10:
        if abs(i - j) == 5:
            return (2 / (-1/2 + math.sqrt(5)/2)) ** 2
        elif abs(i - j) == 4:
            return (2 * math.sqrt(math.sqrt(5)/8 + 5/8) / (-1/2 + math.sqrt(5)/2)) ** 2
        elif abs(i - j) == 3:
            return (2 * (1/4 + math.sqrt(5)/4) / (-1/2 + math.sqrt(5)/2)) ** 2
        elif abs(i - j) == 2:
            return (2 * math.sqrt(5/8 - math.sqrt(5)/8) / (-1/2 + math.sqrt(5)/2)) ** 2
        else:
            return 1

def calcular_incidencias(nv, segmentos):
    incidencias = [[] for _ in range(nv)]
    for idx, (i, j, _) in enumerate(segmentos):
        incidencias[i].append(idx)
        incidencias[j].append(idx)
    return incidencias

def gradiente(vc, segmentos, incidencias):
    grad = np.zeros_like(vc)
    for idx, (i, j, d2_valor) in enumerate(segmentos):
        dist = np.sqrt(d2(vc, i, j))
        if dist == 0:
            continue
        grad[i] += 2 * (dist - math.sqrt(d2_valor)) * (vc[i] - vc[j]) / dist
        grad[j] += 2 * (dist - math.sqrt(d2_valor)) * (vc[j] - vc[i]) / dist
    return grad

class GerarSolidoOperator(bpy.types.Operator):
    bl_idname = "object.gerar_solido_popup"
    bl_label = "Gerar Sólido por Nome"
    bl_options = {'REGISTER', 'UNDO'}

    nome_solido: bpy.props.StringProperty(name="Nome do Sólido", default="J84")

    def execute(self, context):
        pasta = r'C:\Users\thoma\OneDrive\Área de Trabalho\ic\Blender'
        caminho_arquivo = os.path.join(pasta, "faces.txt")
        solidos = carregar_solidos(caminho_arquivo)

        if self.nome_solido not in solidos:
            self.report({'ERROR'}, f"Sólido '{self.nome_solido}' não encontrado.")
            return {'CANCELLED'}

        faces_originais, n = solidos[self.nome_solido]
        faces = corrigir_orientacao_faces(faces_originais)

        edge_set = set()
        edges = []
        for face in faces:
            for i in range(len(face)):
                a = face[i]
                b = face[(i+1) % len(face)]
                if (a, b) not in edge_set and (b, a) not in edge_set:
                    edges.append([a, b])
                    edge_set.add((a, b))

        num_vertices = max(max(f) for f in faces) + 1
        vc = np.random.uniform(-1, 1, (num_vertices, 3))
        segmentos = []
        for f in faces:
            tamanho = len(f)
            for i in range(tamanho):
                for j in range(i + 1, tamanho):
                    d2_valor = calcular_d2(tamanho, f[i], f[j])
                    segmentos.append([f[i], f[j], d2_valor])

        incidencias = calcular_incidencias(num_vertices, segmentos)
        velocity = np.zeros_like(vc)
        alpha = 0.01
        tolerance = 1e-16
        max_iter = 1000000
        norma_grad = 1.0
        i = 0
        ultima_atualizacao = 0

        while norma_grad > tolerance and i < max_iter:
            i += 1
            grad = gradiente(vc, segmentos, incidencias)
            norma_grad = np.linalg.norm(grad)
            velocity -= alpha * grad
            velocity *= 0.99
            vc += velocity

            # Atualização visual otimizada
            if i % 1 == 0 or i == max_iter or norma_grad <= tolerance:
                atualizar_objeto(vc, edges, faces)
                bpy.context.view_layer.update()
                ultima_atualizacao = i

          # Garantir última atualização
        atualizar_objeto(vc, edges, faces)
        obj = bpy.context.active_object
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.context.view_layer.update()

        # Exportação corrigida
        nome_arquivo = os.path.join(pasta, f"{self.nome_solido}.obj")
        bpy.ops.export_scene.obj(
            filepath=nome_arquivo,
            use_selection=False,
            use_active_collection=True,
            global_scale=1.0,
            use_mesh_modifiers=True,
            use_edges=True,
            use_smooth_groups=True,
            use_normals=True,
            use_uvs=True,
            use_materials=True,
            keep_vertex_order=True
        )

        self.report({'INFO'}, f"Sólido exportado para {nome_arquivo}")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

def menu_func(self, context):
    self.layout.operator(GerarSolidoOperator.bl_idname)

def register():
    bpy.utils.register_class(GerarSolidoOperator)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(GerarSolidoOperator)
    bpy.types.VIEW3D_MT_object.remove(menu_func)

if __name__ == "__main__":
    register()