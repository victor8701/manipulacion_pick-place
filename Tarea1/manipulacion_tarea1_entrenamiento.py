#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tarea 1 - Entrenamiento: Evaluacion de todos los agarres

Script auxiliar que prueba los 10 agarres disponibles del YAML,
ejecuta la secuencia pick & place con cada uno, y muestra un
ranking al final para determinar cual agarre da mejor resultado.

Uso:
  1) roslaunch manipulacion_pkg gripper_gazebo.launch tipo_gripper:=jaco objeto:=banana
  2) python3 manipulacion_tarea1_entrenamiento.py
"""

import rospy
import rospkg
import yaml
import PyKDL
import os
import math
import manipulacion_lib
from manipulacion_lib import DetectorColisionesGripperFlotante, Obstaculo
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

# ---------------------------------------------------------------------------
#  PARAMETROS DE CONFIGURACION
# ---------------------------------------------------------------------------
MESA_X = 0.75               # coordenada X objetivo para depositar
MESA_Y = 0.0                # coordenada Y objetivo para depositar
SUPERFICIE_MESA = 0.4       # altura Z de la superficie de la mesa
DIST_SEGURIDAD = 0.15       # margen de seguridad sobre la mesa durante el transporte
DIST_SOLTAR = 0.05          # altura sobre la mesa a la que se suelta el objeto
OFFSET_PREAGARRE = 0.20     # distancia sobre el objeto para la pose de aproximacion
UMBRAL_CAIDA = 0.30         # umbral para considerar que el objeto se ha soltado
PASOS_DESCENSO = 50         # pasos de interpolacion en el descenso al objeto
PASOS_ASCENSO = 50          # pasos de interpolacion en el ascenso tras agarrar
PASOS_SUBIDA_Z = 80         # pasos de interpolacion en la subida a altura de transporte
PASOS_TRANSPORTE_XY = 150   # pasos de interpolacion en el movimiento horizontal
PASOS_DESCENSO_MESA = 30    # pasos de interpolacion en el descenso sobre la mesa
# ---------------------------------------------------------------------------


def resetear_objeto(nombre_objeto, pose_inicial_obj):
    """Devuelve el objeto a su posicion original mediante el servicio de Gazebo."""
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state = ModelState()
        state.model_name = nombre_objeto
        state.pose.position.x = pose_inicial_obj.p[0]
        state.pose.position.y = pose_inicial_obj.p[1]
        state.pose.position.z = pose_inicial_obj.p[2]
        quat = pose_inicial_obj.M.GetQuaternion()
        state.pose.orientation.x = quat[0]
        state.pose.orientation.y = quat[1]
        state.pose.orientation.z = quat[2]
        state.pose.orientation.w = quat[3]
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0
        state.reference_frame = 'world'
        set_state(state)
    except rospy.ServiceException as e:
        print(f"  [WARN] Error al resetear objeto: {e}")


def main():
    # --- Inicializacion ---
    print("=" * 60)
    print("TAREA 1 - ENTRENAMIENTO: EVALUACION DE AGARRES")
    print("=" * 60)

    rospy.init_node('manipulacion_tarea1_entrenamiento', anonymous=True)

    sim = manipulacion_lib.SimulacionGripperFlotante(nombre_gripper_gazebo="gripper")
    sim.configurar_gripper()
    rospy.sleep(2.0)
    sim.abrir_gripper()
    rospy.sleep(3.0)

    pose_inicial = PyKDL.Frame(PyKDL.Rotation(), PyKDL.Vector(0, 0, 0.5))
    sim.fijar_pose_gripper(pose_inicial)
    rospy.sleep(2.0)

    pose_inicial_objeto = sim.obtener_pose_objeto('banana')
    print(f"Posicion inicial banana: [{pose_inicial_objeto.p[0]:.3f}, "
          f"{pose_inicial_objeto.p[1]:.3f}, {pose_inicial_objeto.p[2]:.3f}]")

    # --- Cargar poses de agarre ---
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('manipulacion_pkg')
    yaml_path = os.path.join(pkg_path, 'scripts/grasp_poses/grasp_poses_jaco_banana.yaml')

    with open(yaml_path, 'r') as f:
        grasps = yaml.safe_load(f)['grasps']

    grasps = sorted(grasps, key=lambda g: g['epsilon_quality'], reverse=True)
    print(f"Cargados {len(grasps)} agarres (ordenados por epsilon_quality)")

    # --- Detector de colisiones ---
    obstaculo_suelo = Obstaculo(
        nombre="suelo", tipo="cubo",
        dimensiones=[4.0, 4.0, 0.01],
        pose=[0, 0, -0.005, 0, 0, 0, 1]
    )
    detector = DetectorColisionesGripperFlotante('jaco', [obstaculo_suelo])

    # --- Alturas derivadas ---
    altura_transporte = SUPERFICIE_MESA + DIST_SEGURIDAD + DIST_SOLTAR
    altura_soltar_z = SUPERFICIE_MESA + DIST_SOLTAR
    tipo = sim.get_tipo_gripper()

    # --- Bucle de evaluacion ---
    resultados = []

    print(f"\nParametros: mesa_x={MESA_X}, mesa_y={MESA_Y}, "
          f"superficie={SUPERFICIE_MESA}, seguridad={DIST_SEGURIDAD}, soltar={DIST_SOLTAR}")
    print("-" * 60)

    for i, grasp in enumerate(grasps):
        print(f"\n--- Agarre {i+1}/{len(grasps)} | epsilon={grasp['epsilon_quality']:.5f} ---")

        resultado = {
            'indice': i + 1,
            'epsilon': grasp['epsilon_quality'],
            'colision': False,
            'pick_ok': False,
            'transporte_ok': False,
            'place_ok': False,
            'pos_final': [0, 0, 0],
            'dist_a_objetivo': float('inf'),
            'en_mesa': False,
            'fase_fallo': '',
            'puntuacion': 0.0,
        }

        # Resetear estado
        sim.abrir_gripper()
        rospy.sleep(2.0)
        sim.fijar_pose_gripper(pose_inicial)
        rospy.sleep(1.0)

        for _ in range(5):
            resetear_objeto('banana', pose_inicial_objeto)
            rospy.sleep(1.5)
            pos_check = sim.obtener_pose_objeto('banana').p
            if (abs(pos_check[0] - pose_inicial_objeto.p[0]) < 0.05 and
                abs(pos_check[1] - pose_inicial_objeto.p[1]) < 0.05 and
                abs(pos_check[2] - pose_inicial_objeto.p[2]) < 0.05):
                break
        rospy.sleep(1.0)

        # Calcular poses de agarre en coordenadas mundo
        p = grasp['pose']
        pose_gripper_objeto = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(p[3], p[4], p[5], p[6]),
            PyKDL.Vector(p[0], p[1], p[2])
        )
        pose_objeto_world = sim.obtener_pose_objeto('banana')
        pose_agarre_world = pose_objeto_world * pose_gripper_objeto

        vec_sobre = PyKDL.Vector(
            pose_objeto_world.p[0],
            pose_objeto_world.p[1],
            pose_objeto_world.p[2] + OFFSET_PREAGARRE
        )
        pose_sobre_objeto = PyKDL.Frame(pose_agarre_world.M, vec_sobre)

        # Verificar colisiones
        quat_ag = pose_agarre_world.M.GetQuaternion()
        pose_ag_list = [pose_agarre_world.p[0], pose_agarre_world.p[1], pose_agarre_world.p[2],
                        quat_ag[0], quat_ag[1], quat_ag[2], quat_ag[3]]
        quat_so = pose_sobre_objeto.M.GetQuaternion()
        pose_so_list = [pose_sobre_objeto.p[0], pose_sobre_objeto.p[1], pose_sobre_objeto.p[2],
                        quat_so[0], quat_so[1], quat_so[2], quat_so[3]]

        if detector.hay_colision(pose_ag_list) or detector.hay_colision(pose_so_list):
            print("  Colision detectada, descartado")
            resultado['colision'] = True
            resultado['fase_fallo'] = 'colision'
            resultados.append(resultado)
            continue

        # --- PICK ---
        sim.fijar_pose_gripper(pose_sobre_objeto)
        rospy.sleep(1.5)

        for k in range(PASOS_DESCENSO):
            alpha = (k + 1) / PASOS_DESCENSO
            v = pose_sobre_objeto.p * (1 - alpha) + pose_agarre_world.p * alpha
            sim.fijar_pose_gripper(PyKDL.Frame(pose_agarre_world.M, v))
            rospy.sleep(0.05)

        if tipo == 'jaco':
            sim.set_posicion_articulaciones([-0.6, -0.6, -0.6])
        rospy.sleep(3.0)

        # Ascenso solo en Z
        ag_x = pose_agarre_world.p[0]
        ag_y = pose_agarre_world.p[1]
        ag_z = pose_agarre_world.p[2]
        z_arriba = pose_sobre_objeto.p[2]

        for k in range(PASOS_ASCENSO):
            alpha = (k + 1) / PASOS_ASCENSO
            z_i = ag_z * (1 - alpha) + z_arriba * alpha
            sim.fijar_pose_gripper(PyKDL.Frame(pose_agarre_world.M,
                                               PyKDL.Vector(ag_x, ag_y, z_i)))
            rospy.sleep(0.05)
        rospy.sleep(0.5)

        # Comprobar si se levanto
        alturas = [sim.obtener_pose_objeto('banana').p.z() for _ in range(3)]
        rospy.sleep(0.2)
        alt_min = min(alturas)

        if alt_min < 0.015:
            print(f"  Pick fallido (z={alt_min:.3f})")
            resultado['fase_fallo'] = 'pick fallido'
            resultados.append(resultado)
            continue

        resultado['pick_ok'] = True
        print(f"  Pick correcto (z={alt_min:.3f})")

        # --- TRANSPORTE ---
        rot_ag = pose_agarre_world.M

        # Medir offset del objeto respecto al gripper para compensar el destino
        pos_obj = sim.obtener_pose_objeto('banana').p
        off_x = pos_obj[0] - ag_x
        off_y = pos_obj[1] - ag_y
        off_z = pos_obj[2] - z_arriba

        dest_x = MESA_X - off_x
        dest_y = MESA_Y - off_y
        soltar_z = altura_soltar_z - off_z
        transp_z = soltar_z + DIST_SEGURIDAD

        # Subir en Z
        for k in range(PASOS_SUBIDA_Z):
            alpha = (k + 1) / PASOS_SUBIDA_Z
            z_i = z_arriba * (1 - alpha) + transp_z * alpha
            sim.fijar_pose_gripper(PyKDL.Frame(rot_ag, PyKDL.Vector(ag_x, ag_y, z_i)))
            rospy.sleep(0.05)

        rospy.sleep(0.5)
        if sim.obtener_pose_objeto('banana').p.z() < transp_z - UMBRAL_CAIDA:
            print("  Objeto perdido en subida Z")
            resultado['fase_fallo'] = 'transporte (subida)'
            resultados.append(resultado)
            continue

        # Mover en XY
        perdido = False
        for k in range(PASOS_TRANSPORTE_XY):
            alpha = (k + 1) / PASOS_TRANSPORTE_XY
            x_i = ag_x * (1 - alpha) + dest_x * alpha
            y_i = ag_y * (1 - alpha) + dest_y * alpha
            sim.fijar_pose_gripper(PyKDL.Frame(rot_ag, PyKDL.Vector(x_i, y_i, transp_z)))
            rospy.sleep(0.08)
            if k % 30 == 0 and k > 0:
                if sim.obtener_pose_objeto('banana').p.z() < transp_z - UMBRAL_CAIDA:
                    perdido = True
                    break

        if perdido:
            print("  Objeto perdido en transporte XY")
            resultado['fase_fallo'] = 'transporte (XY)'
            resultados.append(resultado)
            continue

        rospy.sleep(0.3)
        if sim.obtener_pose_objeto('banana').p.z() < transp_z - UMBRAL_CAIDA:
            print("  Objeto perdido tras transporte XY")
            resultado['fase_fallo'] = 'transporte (post-XY)'
            resultados.append(resultado)
            continue

        resultado['transporte_ok'] = True

        # --- PLACE ---
        for k in range(PASOS_DESCENSO_MESA):
            alpha = (k + 1) / PASOS_DESCENSO_MESA
            z_i = transp_z * (1 - alpha) + soltar_z * alpha
            sim.fijar_pose_gripper(PyKDL.Frame(rot_ag, PyKDL.Vector(dest_x, dest_y, z_i)))
            rospy.sleep(0.05)

        sim.abrir_gripper()
        rospy.sleep(2.0)

        pos_banana = sim.obtener_pose_objeto('banana').p
        resultado['pos_final'] = [pos_banana[0], pos_banana[1], pos_banana[2]]

        dx = pos_banana[0] - MESA_X
        dy = pos_banana[1] - MESA_Y
        dist_xy = math.sqrt(dx*dx + dy*dy)
        resultado['dist_a_objetivo'] = dist_xy

        en_mesa = (0.55 <= pos_banana[0] <= 0.95 and
                   -0.55 <= pos_banana[1] <= 0.55 and
                   pos_banana[2] > SUPERFICIE_MESA - 0.10)
        resultado['en_mesa'] = en_mesa
        resultado['place_ok'] = True

        if en_mesa:
            print(f"  Depositado en mesa: [{pos_banana[0]:.3f}, {pos_banana[1]:.3f}, "
                  f"{pos_banana[2]:.3f}] (dist={dist_xy:.3f}m)")
        else:
            print(f"  Fuera de mesa: [{pos_banana[0]:.3f}, {pos_banana[1]:.3f}, "
                  f"{pos_banana[2]:.3f}] (dist={dist_xy:.3f}m)")
            resultado['fase_fallo'] = 'fuera de mesa'

        # Retroceder
        for k in range(PASOS_DESCENSO_MESA):
            alpha = (k + 1) / PASOS_DESCENSO_MESA
            z_i = soltar_z * (1 - alpha) + transp_z * alpha
            sim.fijar_pose_gripper(PyKDL.Frame(rot_ag, PyKDL.Vector(dest_x, dest_y, z_i)))
            rospy.sleep(0.05)

        resultados.append(resultado)

    # --- Volver a casa ---
    sim.abrir_gripper()
    sim.fijar_pose_gripper(pose_inicial)
    resetear_objeto('banana', pose_inicial_objeto)
    rospy.sleep(1.0)

    # --- Puntuacion ---
    for r in resultados:
        pts = 0
        if not r['colision']:
            pts += 10
        if r['pick_ok']:
            pts += 20
        if r['transporte_ok']:
            pts += 20
        if r['place_ok']:
            pts += 20
        if r['en_mesa']:
            pts += 20
            pts += max(0, 10 * (1 - r['dist_a_objetivo'] / 0.3))
        r['puntuacion'] = pts

    ranking = sorted(resultados, key=lambda r: r['puntuacion'], reverse=True)

    # --- Tabla de resultados ---
    print("\n" + "=" * 75)
    print("  RANKING FINAL DE AGARRES")
    print("=" * 75)
    print(f"{'Rank':>4} {'#':>3} {'epsilon':>9} {'Pts':>5} {'Col':>4} "
          f"{'Pick':>5} {'Trans':>5} {'Place':>5} {'Mesa':>5} {'Dist':>6}  {'Estado'}")
    print("-" * 75)

    for rank, r in enumerate(ranking):
        col = "X" if r['colision'] else "ok"
        pick = "ok" if r['pick_ok'] else "-"
        trans = "ok" if r['transporte_ok'] else "-"
        place = "ok" if r['place_ok'] else "-"
        mesa = "ok" if r['en_mesa'] else "-"
        dist = f"{r['dist_a_objetivo']:.3f}" if r['dist_a_objetivo'] < float('inf') else "-"
        fallo = r['fase_fallo'] if r['fase_fallo'] else "OK"
        marca = " <-- MEJOR" if rank == 0 else ""
        print(f"{rank+1:>4} {r['indice']:>3} {r['epsilon']:>9.5f} {r['puntuacion']:>5.1f} "
              f"{col:>4} {pick:>5} {trans:>5} {place:>5} {mesa:>5} {dist:>6}  {fallo}{marca}")

    print("-" * 75)

    mejor = ranking[0]
    print(f"\nMejor agarre: #{mejor['indice']} (epsilon={mejor['epsilon']:.5f}, "
          f"{mejor['puntuacion']:.1f} pts)")
    if mejor['en_mesa']:
        print(f"  Posicion final: [{mejor['pos_final'][0]:.3f}, "
              f"{mejor['pos_final'][1]:.3f}, {mejor['pos_final'][2]:.3f}]")
        print(f"  Distancia al objetivo: {mejor['dist_a_objetivo']:.3f}m")

    n_ok = sum(1 for r in resultados if r['en_mesa'])
    n_pick = sum(1 for r in resultados if r['pick_ok'])
    n_col = sum(1 for r in resultados if r['colision'])
    print(f"\nResumen: {n_ok}/{len(grasps)} completos, {n_pick}/{len(grasps)} pick OK, "
          f"{n_col}/{len(grasps)} con colision")
    print("=" * 75)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("\nProceso interrumpido por el usuario")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
