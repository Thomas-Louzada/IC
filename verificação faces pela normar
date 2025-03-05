import re
import ast

def verificar_orientacao_referencia(face_ref, face):
    intersecao = set(face_ref) & set(face)
    
    if len(intersecao) < 2:
        return True  
    
    v1, v2 = list(intersecao)[:2]
    
    idx1_ref = face_ref.index(v1)
    idx2_ref = face_ref.index(v2)
    
    idx1_face = face.index(v1)
    idx2_face = face.index(v2)
    
    return (idx2_ref - idx1_ref) * (idx2_face - idx1_face) > 0

def inverter_face(face):
    return face[::-1]

def processar_faces(arquivo_entrada, arquivo_saida):
    with open(arquivo_entrada, "r", encoding="utf-8") as f:
        linhas = f.readlines()
    
    faces_corrigidas = []

    for linha in linhas:
        match = re.match(r"(J\d+) - (\[.*\])\s*\|\s*n=(\d+)", linha)
        if match:
            id_solido, faces_str, n_vertices = match.groups()

            # Converte a string de lista para uma estrutura real de Python
            faces = ast.literal_eval(faces_str)
            n_vertices = int(n_vertices)

            face_referencia = faces[0]

            faces_corrigidas_lista = [face_referencia] 
            
            for face in faces[1:]:
                if not verificar_orientacao_referencia(face_referencia, face):
                    face = inverter_face(face)
                faces_corrigidas_lista.append(face)

            # Salva as faces corrigidas
            faces_corrigidas.append(f"{id_solido} - {str(faces_corrigidas_lista)} | n={n_vertices}\n")

    with open(arquivo_saida, "w", encoding="utf-8") as f_out:
        f_out.writelines(faces_corrigidas)

arquivo_entrada = "faces.txt"   
arquivo_saida = "faces_corrigidas.txt"  
processar_faces(arquivo_entrada, arquivo_saida)

print(f"Arquivo {arquivo_saida} gerado com sucesso!")
