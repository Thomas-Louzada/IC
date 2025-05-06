bl_info = {
    "name": "Gerador de Sólidos Automatizado (Animação e Vídeo via Câmera)",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import os
import ast
import numpy as np
import math
from collections import defaultdict, deque

# --------------------
# Helpers
# --------------------

def carregar_solidos(arquivo):
    solidos = {}
    if not os.path.isfile(arquivo):
        raise FileNotFoundError(f"Arquivo de faces não encontrado: {arquivo}")
    with open(arquivo, 'r') as f:
        for linha in f:
            if " | n=" in linha:
                nome_e_faces, n_str = linha.split(" | n=")
                nome, faces_raw = nome_e_faces.split(" - ")
                faces = ast.literal_eval(faces_raw)
                solidos[nome.strip()] = (faces, int(n_str.strip()))
    return solidos


def corrigir_orientacao_faces(faces):
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        m = len(face)
        for i in range(m):
            key = tuple(sorted((face[i], face[(i + 1) % m])))
            edge_to_faces[key].append(fi)

    oriented = [None] * len(faces)
    processed = [False] * len(faces)
    oriented[0] = list(faces[0])
    processed[0] = True
    queue = deque([0])
    while queue:
        fi = queue.popleft()
        ref = oriented[fi]
        for i in range(len(ref)):
            a, b = ref[i], ref[(i + 1) % len(ref)]
            key = tuple(sorted((a, b)))
            for nbr in edge_to_faces[key]:
                if not processed[nbr]:
                    f2 = list(faces[nbr])
                    for k in range(len(f2)):
                        x, y = f2[k], f2[(k + 1) % len(f2)]
                        if {x, y} == {a, b}:
                            if (x, y) == (b, a):
                                f2.reverse()
                            break
                    oriented[nbr] = f2
                    processed[nbr] = True
                    queue.append(nbr)
    for i, f in enumerate(oriented):
        if f is None:
            oriented[i] = list(faces[i])
    return oriented


def calcular_d2(n, i, j):
    # Distância ao quadrado baseada no polígono regular de n vértices
    if n == 3:
        return 1
    elif n == 4:
        return math.sqrt(2)**2 if abs(i - j) == 2 else 1
    elif n == 5:
        return ((1 + math.sqrt(5)) / 2)**2 if abs(i - j) in [2, 3] else 1
    elif n == 6:
        return (math.sqrt(3))**2 if abs(i - j) == 3 else (4 if abs(i - j) == 2 else 1)
    elif n == 8:
        if abs(i - j) == 4:
            return (1 / math.sqrt(1/2 - math.sqrt(2)/4))**2
        elif abs(i - j) == 3:
            return (math.sqrt(math.sqrt(2)/4 + 1/2) / math.sqrt(1/2 - math.sqrt(2)/4))**2
        elif abs(i - j) == 2:
            return (math.sqrt(2) / (2*math.sqrt(1/2 - math.sqrt(2)/4)))**2
        else:
            return 1
    elif n == 10:
        if abs(i - j) == 5:
            return (2 / (-1/2 + math.sqrt(5)/2))**2
        elif abs(i - j) == 4:
            return (2*math.sqrt(math.sqrt(5)/8 + 5/8) / (-1/2 + math.sqrt(5)/2))**2
        elif abs(i - j) == 3:
            return (2*(1/4 + math.sqrt(5)/4) / (-1/2 + math.sqrt(5)/2))**2
        elif abs(i - j) == 2:
            return (2*math.sqrt(5/8 - math.sqrt(5)/8) / (-1/2 + math.sqrt(5)/2))**2
        else:
            return 1


def gerar_segmentos(faces):
    """
    Gera segmentos com distâncias corretas para cada par de vértices em cada face,
    usando índices locais (i, j) para cálculo de distância quadrada.
    """
    segs = []
    for f in faces:
        m = len(f)
        for i in range(m):
            for j in range(i + 1, m):
                # usa índices locais i, j no cálculo de distância do polígono regular
                d2 = calcular_d2(m, i, j)
                # f[i], f[j] são os índices globais dos vértices na malha
                segs.append((f[i], f[j], d2))
    return np.array(segs, dtype=float)


def gradiente(vc, segmentos):
    grad = np.zeros_like(vc)
    for seg in segmentos:
        i, j, d2 = int(seg[0]), int(seg[1]), seg[2]
        diff = vc[i] - vc[j]
        dist = np.linalg.norm(diff)
        if dist > 0:
            factor = 2*(dist - math.sqrt(d2))/dist
            grad[i] += factor*diff
            grad[j] -= factor*diff
    return grad


def atualizar_mesh(obj, vc):
    m = obj.data
    m.vertices.foreach_set('co', vc.ravel())
    m.update()

# --------------------
# Operator Principal
# --------------------

class GerarSolidoOperator(bpy.types.Operator):
    bl_idname = 'object.gerar_solido_popup'
    bl_label = 'Gerar, Animar e Gravar (Câmera)'
    bl_options = {'REGISTER'}

    faces_path: bpy.props.StringProperty(subtype='FILE_PATH', default='')
    nome_solido: bpy.props.StringProperty(default='')
    video_path: bpy.props.StringProperty(subtype='FILE_PATH', default='')
    tol_exp: bpy.props.IntProperty(default=10, min=0)
    alpha: bpy.props.FloatProperty(default=0.01, min=0.0)
    max_iter: bpy.props.IntProperty(default=10000, min=1)
    angle_step: bpy.props.FloatProperty(
        name="Grau por 10 iterações",
        default=0.0, min=0.0, max=360.0
    )

    _timer = None
    _vc = None
    _vel = None
    _segments = None
    _edges = None
    _faces = None
    _obj = None
    _it = 0
    _tol = 0.0
    _frame = 1
    _angle_rad = 0.0
    _target = None

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'faces_path')
        layout.prop(self, 'nome_solido')
        layout.prop(self, 'video_path')
        layout.prop(self, 'tol_exp')
        layout.prop(self, 'alpha')
        layout.prop(self, 'max_iter')
        layout.prop(self, 'angle_step')

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        # Inicialização dos parâmetros
        self._tol = 10**(-self.tol_exp)
        self._angle_rad = math.radians(self.angle_step)

        # Carregamento e correção das faces
        fpath = bpy.path.abspath(self.faces_path) if self.faces_path else os.path.join(os.path.dirname(__file__), 'faces.txt')
        solidos = carregar_solidos(fpath)
        if self.nome_solido not in solidos:
            self.report({'ERROR'}, 'Sólido não encontrado')
            return {'CANCELLED'}
        f0, _ = solidos[self.nome_solido]
        self._faces = corrigir_orientacao_faces(f0)

        # Geração de arestas e segmentos
        s = set()
        self._edges = []
        for face in self._faces:
            for i in range(len(face)):
                a, b = face[i], face[(i+1)%len(face)]
                if (a, b) not in s:
                    self._edges.append((a, b))
                    s.update({(a, b), (b, a)})
        self._segments = gerar_segmentos(self._faces)

        # Inicialização dos vértices e velocidades
        nv = int(self._segments[:,:2].max()) + 1
        self._vc = np.random.uniform(-1, 1, (nv, 3))
        self._vel = np.zeros_like(self._vc)

        # Criação do mesh
        mesh = bpy.data.meshes.new(f"sol_{self.nome_solido}")
        self._obj = bpy.data.objects.new(f"sol_{self.nome_solido}", mesh)
        context.collection.objects.link(self._obj)
        mesh.from_pydata(self._vc.tolist(), self._edges, self._faces)
        mesh.update()

        # Cria Empty para câmera e aplica constraint
        empty = bpy.data.objects.new("Camera_Target", None)
        empty.location = self._obj.location
        context.collection.objects.link(empty)
        self._target = empty
        cam = context.scene.camera or bpy.data.objects.new("Camera_Solido", bpy.data.cameras.new("Cam"))
        if not context.scene.camera:
            context.collection.objects.link(cam)
            context.scene.camera = cam
        cam.parent = empty
        cam.location = (0, -5, 0)
        tr = cam.constraints.new('TRACK_TO')
        tr.target = empty
        tr.track_axis = 'TRACK_NEGATIVE_Z'
        tr.up_axis = 'UP_Y'

        # Inicia modal
        self._it = 0
        self._frame = 1
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        print(f"Iniciando geração de '{self.nome_solido}' com tol=1e-{self.tol_exp}, rot={self.angle_step}°/10 it.")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            for _ in range(10):
                grad = gradiente(self._vc, self._segments)
                self._it += 1
                if self._it >= self.max_iter or np.linalg.norm(grad) <= self._tol:
                    atualizar_mesh(self._obj, self._vc)
                    context.scene.frame_current = self._frame
                    return self.finish(context)
                self._vel = self._vel * 0.99 - self.alpha * grad
                self._vc += self._vel
            atualizar_mesh(self._obj, self._vc)
            context.scene.frame_current = self._frame
            self._frame += 1
            # orbita e keyframe
            self._target.rotation_euler[2] += self._angle_rad
            self._target.keyframe_insert(data_path='rotation_euler', frame=self._frame)
            print(f"Iteração {self._it}: rotação {math.degrees(self._target.rotation_euler[2])%360:.1f}°")
        return {'PASS_THROUGH'}

    def finish(self, context):
        print(f"Finalizado: {self._it} iterações, {self._frame-1} quadros.")
        context.scene.frame_start = 1
        context.scene.frame_end = self._frame
        if self.video_path:
            context.scene.render.filepath = bpy.path.abspath(self.video_path)
            context.scene.render.image_settings.file_format = 'FFMPEG'
            context.scene.render.ffmpeg.format = 'MPEG4'
            context.scene.render.ffmpeg.codec = 'H264'
            context.scene.render.ffmpeg.constant_rate_factor = 'HIGH'
            bpy.ops.render.render(animation=True, use_viewport=True)
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        self.report({'INFO'}, f"'{self.nome_solido}' gerado e animado com sucesso.")
        return {'FINISHED'}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
        self.report({'WARNING'}, 'Execução cancelada pelo usuário.')
        return {'CANCELLED'}

# --------------------
# Registro do Addon
# --------------------

def menu_func(self, context):
    self.layout.operator(GerarSolidoOperator.bl_idname)

def register():
    bpy.utils.register_class(GerarSolidoOperator)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(GerarSolidoOperator)
    bpy.types.VIEW3D_MT_object.remove(menu_func)

if __name__ == '__main__':
    register()
