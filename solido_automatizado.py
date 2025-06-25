bl_info = {
    "name": "Gerador de Sólidos Automatizado (Animação e Vídeo via Câmera)",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import os
import ast
import json
import numpy as np
import math
import shutil
import glob
import time
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
            key = tuple(sorted((face[i], face[(i+1) % m])))
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
            a, b = ref[i], ref[(i+1) % len(ref)]
            key = tuple(sorted((a, b)))
            for nbr in edge_to_faces[key]:
                if not processed[nbr]:
                    f2 = list(faces[nbr])
                    for k in range(len(f2)):
                        x, y = f2[k], f2[(k+1) % len(f2)]
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


def gerar_segmentos(faces):
    segs = []
    for f in faces:
        m = len(f)
        for i in range(m):
            for j in range(i+1, m):
                d2 = calcular_d2(m, i, j)
                segs.append((f[i], f[j], d2))
    return np.array(segs, dtype=float)


def gradiente(vc, segmentos):
    grad = np.zeros_like(vc)
    for seg in segmentos:
        i, j, d2 = int(seg[0]), int(seg[1]), seg[2]
        diff = vc[i] - vc[j]
        dist = np.linalg.norm(diff)
        if dist > 0:
            fator = 2 * (dist - math.sqrt(d2)) / dist
            grad[i] += fator * diff
            grad[j] -= fator * diff
    return grad


def compute_distance_signature(vc):
    centroid = vc.mean(axis=0)
    y = vc - centroid
    n = len(y)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(y[i] - y[j])
            dists.append(round(d, 6))
    return tuple(sorted(dists))


def atualizar_mesh(obj, vc):
    mesh = obj.data
    mesh.vertices.foreach_set('co', vc.ravel())
    mesh.update()


class GerarSolidoOperator(bpy.types.Operator):
    bl_idname = 'object.gerar_solido_popup'
    bl_label = 'Gerar, Animar e Gravar (Câmera)'
    bl_options = {'REGISTER'}

    faces_path: bpy.props.StringProperty(subtype='FILE_PATH')
    nome_solido: bpy.props.StringProperty()
    video_path: bpy.props.StringProperty(subtype='FILE_PATH')
    catalog_path: bpy.props.StringProperty(
        name="Caminho do Catalog.json",
        description="Local onde o arquivo de catálogo será salvo",
        subtype='FILE_PATH'
    )
    tol_exp: bpy.props.IntProperty(default=10, min=0)
    alpha: bpy.props.FloatProperty(default=0.01, min=0.0)
    max_iter: bpy.props.IntProperty(default=10000, min=1)
    angle_step: bpy.props.FloatProperty(name="Grau por 10 iterações", default=0.0, min=0.0, max=360.0)

    _timer = None
    _vc = None
    _vel = None
    _segments = None
    _faces = None
    _obj = None
    _it = 0
    _tol = 0.0
    _frame = 1
    _angle_rad = 0.0
    _target = None
    _out_dir = None
    _stopped_by_tol = False
    seen_signatures = set()

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'faces_path')
        layout.prop(self, 'nome_solido')
        layout.prop(self, 'video_path')
        layout.prop(self, 'catalog_path')
        layout.prop(self, 'tol_exp')
        layout.prop(self, 'alpha')
        layout.prop(self, 'max_iter')
        layout.prop(self, 'angle_step')

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        scene = context.scene
        if scene.sequence_editor:
            scene.sequence_editor_clear()

        timestamp = int(time.time())
        self._out_dir = bpy.path.abspath(self.video_path) + f'_{self.nome_solido}_{timestamp}'
        os.makedirs(self._out_dir, exist_ok=True)

        # Usa catalog_path se fornecido, caso contrário salva no addon
        if self.catalog_path:
            self._catalog_file = bpy.path.abspath(self.catalog_path)
        else:
            addon_dir = os.path.dirname(__file__)
            self._catalog_file = os.path.join(addon_dir, "catalog.json")

        # Carrega catálogo prévio, se existir
        if os.path.exists(self._catalog_file):
            with open(self._catalog_file, 'r') as f:
                GerarSolidoOperator.seen_signatures = set(json.load(f))

        scene.render.image_settings.file_format = 'PNG'

        fpath = bpy.path.abspath(self.faces_path) if self.faces_path else os.path.join(os.path.dirname(__file__), 'faces.txt')
        sol = carregar_solidos(fpath)
        if self.nome_solido not in sol:
            self.report({'ERROR'}, 'Sólido não encontrado no arquivo')
            return {'CANCELLED'}
        faces0, _ = sol[self.nome_solido]
        self._faces = corrigir_orientacao_faces(faces0)

        self._segments = gerar_segmentos(self._faces)
        edges, seen = [], set()
        for face in self._faces:
            for i in range(len(face)):
                a, b = face[i], face[(i+1) % len(face)]
                if (a,b) not in seen:
                    edges.append((a,b)); seen.update({(a,b),(b,a)})

        nv = int(self._segments[:, :2].max()) + 1
        self._vc = np.random.uniform(-1, 1, (nv, 3))
        self._vel = np.zeros_like(self._vc)

        mesh = bpy.data.meshes.new(f"sol_{self.nome_solido}")
        self._obj = bpy.data.objects.new(f"sol_{self.nome_solido}", mesh)
        context.collection.objects.link(self._obj)
        mesh.from_pydata(self._vc.tolist(), edges, self._faces)
        mesh.update()

        empty = bpy.data.objects.new("Camera_Target", None)
        empty.location = self._obj.location
        context.collection.objects.link(empty)
        self._target = empty
        cam = context.scene.camera or bpy.data.objects.new("Camera_Solido", bpy.data.cameras.new("Cam"))
        if not context.scene.camera:
            context.collection.objects.link(cam)
            context.scene.camera = cam
        cam.parent = empty; cam.location = (0, -5, 0)
        tr = cam.constraints.new('TRACK_TO'); tr.target = empty
        tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'

        self._it = 0; self._frame = 1; self._angle_rad = 0.0
        self._tol = 10 ** (-self.tol_exp)
        self._stopped_by_tol = False

        wm = context.window_manager
        wm.progress_begin(0, self.max_iter)
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            for _ in range(10):
                g = gradiente(self._vc, self._segments)
                grad_norm = np.linalg.norm(g)
                self._it += 1
                wm = context.window_manager
                wm.progress_update(self._it)
                if self._it >= self.max_iter or grad_norm <= self._tol:
                    self._stopped_by_tol = (grad_norm <= self._tol)
                    atualizar_mesh(self._obj, self._vc)
                    return self.finish(context)
                self._vel = self._vel * 0.99 - self.alpha * g
                self._vc += self._vel

            atualizar_mesh(self._obj, self._vc)
            context.scene.frame_set(self._frame)
            img_path = os.path.join(self._out_dir, f"frame_{self._frame:04d}.png")
            context.scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)

            if self.angle_step > 0.0:
                self._angle_rad += math.radians(self.angle_step)
                self._target.rotation_euler = (0, 0, self._angle_rad)

            self._frame += 1
        return {'PASS_THROUGH'}

    def finish(self, context):
        # ------------------------------------------------------------
        # Verifica se a configuração já existe no catálogo
        # ------------------------------------------------------------
        signature = compute_distance_signature(self._vc)
        if signature in GerarSolidoOperator.seen_signatures:
            self.report({'INFO'}, f"Sólido '{self.nome_solido}' duplicado — descartado")
            with open(self._catalog_file, 'w') as f:
                json.dump(list(GerarSolidoOperator.seen_signatures), f)
            return {'CANCELLED'}
        else:
            GerarSolidoOperator.seen_signatures.add(signature)
            with open(self._catalog_file, 'w') as f:
                json.dump(list(GerarSolidoOperator.seen_signatures), f)
        # ------------------------------------------------------------

        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()

        scene = context.scene
        if not scene.sequence_editor:
            scene.sequence_editor_create()
        seq = scene.sequence_editor.sequences
        files = sorted(glob.glob(os.path.join(self._out_dir, 'frame_*.png')))
        for idx, filepath in enumerate(files, start=1):
            seq.new_image(name=f"Frame{idx:04d}", filepath=filepath, channel=1, frame_start=idx)

        scene.render.use_sequencer = True
        scene.render.filepath = bpy.path.abspath(self.video_path)
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        scene.render.ffmpeg.constant_rate_factor = 'HIGH'
        scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
        scene.frame_start = 1
        scene.frame_end = self._frame - 1

        bpy.ops.render.render(animation=True, use_viewport=False)
        shutil.rmtree(self._out_dir)

        motivo = 'tolerância' if self._stopped_by_tol else 'número máximo de iterações'
        self.report({'INFO'}, f"Concluído em {self._it} iterações. Parado por {motivo}.")
        return {'FINISHED'}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            context.window_manager.progress_end()
        self.report({'WARNING'}, 'Execução cancelada pelo usuário.')
        return {'CANCELLED'}
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