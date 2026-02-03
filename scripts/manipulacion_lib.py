#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 2023
@title: manipulacion_lib.py
@description: Librería de funciones para la asignatura de manipulación
@python_version: 3.8.10
@autor: Elisabeth Menendez Salvador
@contact: emenende@ing.uc3m.es
"""

# manipulacion_lib.py

import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ipywidgets import FloatSlider, Layout, VBox, HTML
from urdf_parser_py.urdf import URDF
from IPython.display import display
import PyKDL as PyKDL
import rospy
import urdf_parser_py.urdf
from kdl_parser_py.urdf import treeFromUrdfModel
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
from std_msgs.msg import ColorRGBA
import tf.transformations as tf_trans
# from scipy.interpolate import CubicSpline
import yaml
import pinocchio as pin
import hppfcl
import rospkg
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
import yaml
from trac_ik_python.trac_ik import IK

def esperar_por_subscribers(publisher, timeout=None):
    """
    Bloquea hasta que el publicador dado tenga al menos un suscriptor.
    Parámetros de entrada:
    - publisher: El publicador de ROS
    - timeout: Tiempo límite opcional en segundos
    Return:
    - True si el publicador tiene suscriptores, False si se agota el tiempo límite

    """
    if timeout is not None:
        timeout_time = rospy.get_time() + timeout
    while publisher.get_num_connections() == 0:
        if timeout is not None and rospy.get_time() > timeout_time:
            return False
        rospy.sleep(0.1)  
    return True


def convert_kdl_frame_to_geometry_pose(kdl_frame):
    # Convert the position (translation) from PyKDL Vector to geometry_msgs/Point
    position = Point()
    position.x = kdl_frame.p[0]
    position.y = kdl_frame.p[1]
    position.z = kdl_frame.p[2]

    # Convert the orientation (rotation) from PyKDL Rotation to geometry_msgs/Quaternion
    rotation = kdl_frame.M.GetQuaternion()  # This returns a tuple (x, y, z, w)
    orientation = Quaternion()
    orientation.x = rotation[0]
    orientation.y = rotation[1]
    orientation.z = rotation[2]
    orientation.w = rotation[3]

    # Create a geometry_msgs/Pose and assign the position and orientation
    pose = Pose()
    pose.position = position
    pose.orientation = orientation

    return pose

def guardar_grasps_en_yaml(agarres, nombre_archivo):
    """
    Guarda los datos de los agarres en un archivo YAML.

    Parámetros de entrada:
    - agarres: Lista de objetos de agarre.
    - nombre_archivo: String, nombre del archivo YAML donde se guardarán los datos.

    Return:
    Nada
    """
    # Initialize the data dictionary
    data = {'grasps': []}
    # Iterate over the grasps and add their information to the data dictionary
    for grasp in agarres:
        grasp_info = {
            'pose': [grasp.pose.position.x, grasp.pose.position.y, grasp.pose.position.z,
                     grasp.pose.orientation.x, grasp.pose.orientation.y, grasp.pose.orientation.z, grasp.pose.orientation.w],
            'dofs': list(grasp.dofs),
            'epsilon_quality': grasp.epsilon_quality,
            'volume_quality': grasp.volume_quality
        }
        data['grasps'].append(grasp_info)

    # Write the data to the YAML file
    with open(nombre_archivo, 'w') as file:
        yaml.dump(data, file)


def extraer_posicion_superficie_plana_de_yaml(yaml_file, nombre_objeto):
    """
    Extrae la posición de cada objeto definido en un archivo YAML.

    Parámetros:
    - yaml_file: Ruta al archivo YAML.
    - nombre_objeto: Nombre de el objeto agarrable

    Retorna:
    - Las posiciones de la superfice plana sobre la que se encuentra el objeto agarrable.
    """
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    positions = []
    for object_name, object_data in data.items():
      if object_name == nombre_objeto:
        positions = [object_data['position']['x'],object_data['position']['y'], object_data['position']['z']]
        orientations =  [object_data['orientation']['x'],object_data['orientation']['y'], object_data['orientation']['z'], object_data['orientation']['w']]
    return positions, orientations



def obtener_nombres_articulaciones_de_urdf():
    """
    Obtiene los nombres de las articulaciones no fijas del robot a partir del URDF.

    Parámetros de entrada:
    No hay parámetros de entrada para esta función.

    Return:
    Retorna una lista con los nombres de las articulaciones del robot (str). 
    Solo incluye las articulaciones que no son de tipo 'fixed'.

    Esta función lee la descripción del robot desde el parámetro '/robot_description' de ROS,
    y luego utiliza URDF para parsear la información del robot.
    """
    descripcion_robot = rospy.get_param('/robot_description')
    robot = URDF.from_xml_string(descripcion_robot)
    return [articulacion.name for articulacion in robot.joints if articulacion.type != 'fixed' 
            and 'hand' not in articulacion.name]


def frame_kdl_a_pose(frame):
    """
    Convierte un Frame KDL en un mensaje Pose de ROS.

    Parámetros de entrada:
    frame: Objeto PyKDL.Frame que incluye la rotación (M) y la posición (p) del frame.

    Return:
    Retorna un objeto Pose de ROS, representando la posición y orientación del Frame.

    Esta función extrae la rotación y la posición del Frame KDL, convierte la rotación en un cuaternión,
    y las combina en un mensaje Pose de ROS.
    """
    
    # Extrae la posición del frame
    position = frame.p
    pose_position = Point(position[0], position[1], position[2])

    # Convierte la rotación (una rotación PyKDL) en un cuaternión
    rotation = frame.M
    quaternion = rotation.GetQuaternion()
    pose_orientation = Quaternion(*quaternion)

    # Crea y retorna el mensaje Pose
    pose = Pose()
    pose.position = pose_position
    pose.orientation = pose_orientation
    return pose

def publicar_marcador(pub_marcador, pose_efector_final, frame_id="base_link"):
    
    """
    Publica un marcador en RViz para visualizar la pose del efector final.

    Parámetros de entrada:
    pub_marcador (rospy.Publisher): Publicador ROS para enviar marcadores.
    pose_efector_final (Pose): Pose del efector final a visualizar.
    frame_id (str): ID del sistema de referencia para el marcador (por defecto 'base_link').

    La función convierte la pose a un Frame PyKDL, extrae los vectores unitarios de los ejes locales,
    crea marcadores para cada eje y los publica en RViz.
    """
        
    # Convert the pose to a PyKDL Frame
    frame = PyKDL.Frame(PyKDL.Rotation.Quaternion(pose_efector_final.orientation.x, pose_efector_final.orientation.y, 
                                                pose_efector_final.orientation.z, pose_efector_final.orientation.w), 
                        PyKDL.Vector(pose_efector_final.position.x, pose_efector_final.position.y, pose_efector_final.position.z))

    # Extract the unit vectors of the local axes
    local_x = frame.M.UnitX()*0.2
    local_y = frame.M.UnitY()*0.2
    local_z = frame.M.UnitZ()*0.2

    # Convert the unit vectors to Points
    start_point = Point(pose_efector_final.position.x, pose_efector_final.position.y, pose_efector_final.position.z)
    end_point_x = Point(start_point.x + local_x[0], start_point.y + local_x[1], start_point.z + local_x[2])
    end_point_y = Point(start_point.x + local_y[0], start_point.y + local_y[1], start_point.z + local_y[2])
    end_point_z = Point(start_point.x + local_z[0], start_point.y + local_z[1], start_point.z + local_z[2])

    # Create markers for each axis
    x_axis_marker = crear_marcador_flecha(1, start_point, end_point_x, ColorRGBA(1, 0, 0, 1))  # Red for X
    y_axis_marker = crear_marcador_flecha(2, start_point, end_point_y, ColorRGBA(0, 1, 0, 1))  # Green for Y
    z_axis_marker = crear_marcador_flecha(3, start_point, end_point_z, ColorRGBA(0, 0, 1, 1))  # Blue for Z
    marker_array = MarkerArray()
    marker_array.markers.append(x_axis_marker)
    marker_array.markers.append(y_axis_marker)
    marker_array.markers.append(z_axis_marker)
    # Create a MarkerArray and add the markers
    
    # Publish the marker
    pub_marcador.publish(marker_array)
    
def crear_marcador_flecha(marker_id: int, punto_inicio: Point, punto_final: Point, color: ColorRGBA, frame_id: str = "base_link")-> Marker:
    """
    Crea un marcador en forma de flecha para RViz.

    Parámetros de entrada:
    marker_id (int): Identificador único del marcador.
    punto_inicio (Point): Punto de inicio de la flecha.
    punto_final (Point): Punto final de la flecha.
    color (ColorRGBA): Color de la flecha.
    frame_id (str): ID del marco de referencia (por defecto 'base_link').

    Return:
    Retorna un objeto Marker configurado como una flecha.

    Define la forma, tamaño, color y orientación de la flecha.
    """
    
    marker = Marker()
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.header.frame_id = frame_id  
    marker.header.stamp = rospy.Time.now()

    # Set the start and end points of the arrow
    marker.points.append(punto_inicio)
    marker.points.append(punto_final)

    # Scale and color
    marker.scale.x = 0.025  # Shaft diameter
    marker.scale.y = 0.05  # Head diameter
    marker.scale.z = 0    # Not used for arrows defined by points
    marker.color = color

    return marker

class Robot:
    def __init__(self, nombres_articulaciones=None):
        """
        Inicializa una instancia de la clase Robot.

        Parámetros de entrada:
        nombres_articulaciones: Lista opcional de nombres de articulaciones. 
                                Si es None, se obtendrán de la descripción URDF.

        Este constructor obtiene la descripción URDF del robot, la parsea,
        y establece los nombres de las articulaciones, sus límites inferiores y superiores,
        y una lista inicial de posiciones de articulaciones.
        """
        
        # Obtener la descripción URDF del servidor de parámetros
        self.descripcion_robot = rospy.get_param('/robot_description')

        # Parsear URDF
        self.robot_urdf = urdf_parser_py.urdf.URDF.from_xml_string(self.descripcion_robot)

        if nombres_articulaciones is None:
            nombres_articulaciones = obtener_nombres_articulaciones_de_urdf()
            
        self.nombres_articulaciones = nombres_articulaciones
        self.limites_inferiores = []
        self.limites_superiores = []
        self.posiciones_articulaciones = [0.0] * len(self.nombres_articulaciones)
        for articulacion in self.nombres_articulaciones:
            limite_inferior, limite_superior = self.obtener_limites_articulacion(articulacion)
            self.limites_inferiores.append(limite_inferior)
            self.limites_superiores.append(limite_superior)
        self.limites_inferioes_kdl_array = PyKDL.JntArray(len(self.limites_inferiores))
        self.limites_superiores_kdl_array = PyKDL.JntArray(len(self.limites_superiores))
        for i, limite_inferior in enumerate(self.limites_inferiores):
            self.limites_inferioes_kdl_array[i] = limite_inferior
            self.limites_superiores_kdl_array[i] = limite_superior

    def obtener_limites_articulacion(self, nombre_articulacion):
        
        """
        Obtiene los límites inferior y superior de una articulación específica.

        Parámetros de entrada:
        nombre_articulacion: El nombre de la articulación para la cual se buscan los límites.

        Return:
        Tuple (limite_inferior, limite_superior) que contiene los límites de la articulación.
        Si la articulación no se encuentra, se lanza un ValueError.

        La función recorre todas las articulaciones definidas en el URDF del robot y compara sus nombres
        con el proporcionado. Cuando encuentra una coincidencia, retorna sus límites.
        """
        
        for articulacion in self.robot_urdf.joints:
            if articulacion.name == nombre_articulacion:
                limite_inferior = articulacion.limit.lower
                limite_superior = articulacion.limit.upper
                return limite_inferior, limite_superior
        raise ValueError(f"Articulacion {nombre_articulacion} no encontrada")
    
    def obtener_nombres_articulaciones(self):
        return self.nombres_articulaciones
    
    def get_limites_inferiores(self):
        return self.limites_inferiores
    
    def get_limites_superiores(self):
        return self.limites_superiores



class FakeRobot(Robot):
    
    def fijar_posicion_articulaciones(self, pub_estados_articulaciones, posiciones_articulaciones):
        """
        Publica las posiciones actuales de las articulaciones del robot.

        Parámetros de entrada:
        pub_estados_articulaciones: Publisher ROS para enviar el estado de las articulaciones.
        posiciones_articulaciones: Lista con las posiciones deseadas para cada articulación.

        Esta función verifica que el número de articulaciones y posiciones proporcionadas coincida.
        Crea un mensaje JointState, asigna los nombres y posiciones de las articulaciones,
        actualiza los valores internos de las articulaciones y publica el mensaje en ROS.
        """
        
        if len(self.nombres_articulaciones) != len(posiciones_articulaciones):
            raise ValueError('El numero de nombres_articulaciones y posiciones_articulaciones deben ser iguales')
        estado_articulacion = JointState()
        estado_articulacion.header.stamp = rospy.Time.now()
        estado_articulacion.name = self.nombres_articulaciones
        estado_articulacion.position = posiciones_articulaciones
        self.posiciones_articulaciones = posiciones_articulaciones
        esperar_por_subscribers(pub_estados_articulaciones, 2)
        pub_estados_articulaciones.publish(estado_articulacion)

    def obtener_posiciones_articulaciones(self):
        """
        Devuelve las posiciones actuales de las articulaciones del robot.

        No requiere parámetros de entrada.

        Return:
        Retorna una lista con los valores actuales de las posiciones de las articulaciones.
        """
        return self.posiciones_articulaciones

class GazeboRobot(Robot):      
    def __init__(self, nombres_articulaciones=None):
        super().__init__(nombres_articulaciones=nombres_articulaciones)   
        self.pub = rospy.Publisher('/eff_joint_traj_controller/command', JointTrajectory, queue_size=10)
    
    def wrap_to_pi(self, q):
        q = np.array(q, dtype=float)
        return (q + np.pi) % (2 * np.pi) - np.pi
    
    def obtener_posiciones_articulaciones(self):
        message = rospy.wait_for_message("joint_states", JointState)
      
        # Filter and print positions of the joints of interest
        ordered_positions = []
      
        # Loop through the list of joints of interest
        for joint_name in self.nombres_articulaciones:
          # Find the index of this joint in the received message
          if joint_name in message.name:
              index = message.name.index(joint_name)
              # Append the corresponding position to the ordered list
              ordered_positions.append(message.position[index])
        ordered_positions = self.wrap_to_pi(ordered_positions)
        # Return or use the joint_positions as needed
        return ordered_positions
    
    def command_posicion_articulaciones(self, posicion_articulaciones, time_from_start):
        trajectory_points = []
        point = JointTrajectoryPoint()
        point.positions = posicion_articulaciones
        point.time_from_start = rospy.Duration(time_from_start)
        trajectory_points.append(point)

        # Create the trajectory message
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint', 
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        trajectory.points = trajectory_points
        # Publish the trajectory
        esperar_por_subscribers(self.pub, 2)
        self.pub.publish(trajectory)
        rospy.sleep(0.01)

    def command_path_posicion_articulaciones(self, path, time_between_points, start_time):
        trajectory_points = []


        for i, p in enumerate(path):
          point = JointTrajectoryPoint()
          point.positions = p
          point.time_from_start = i*rospy.Duration(time_between_points)+rospy.Duration(start_time)
          trajectory_points.append(point)
        
        # Create the trajectory message
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint', 
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        trajectory.points = trajectory_points
        # print(trajectory)
        # Publish the trajectory
        esperar_por_subscribers(self.pub, 2)
        self.pub.publish(trajectory)
        rospy.sleep(time_between_points*len(path)+start_time)

class ConfiguradorSlidersArticulaciones:
    def __init__(self, robot=None, pub_estados_articulaciones=None, nombres_articulaciones=None):
        
        """
        Inicializa un configurador de sliders para las articulaciones del robot.

        Parámetros de entrada:
        robot: Objeto FakeRobot con las articulaciones y sus límites.
        pub_estados_articulaciones: Publicador ROS para enviar los estados de las articulaciones.
        nombres_articulaciones: Nombres de las articulaciones del robot (opcional).

        Crea sliders para controlar las posiciones de las articulaciones y los muestra en pantalla.
        """
        
        self.robot = robot
        self.pub_estados_articulaciones = pub_estados_articulaciones
        self.sliders = self.crear_sliders()
        vbox = VBox([*self.sliders])
        display(vbox)
        for slider in self.sliders:
            slider.observe(self.actualiza_posicion_articulaciones, names='value')

    def publicar_estados_articulaciones(self):
        
        """
        Publica los estados actuales de las articulaciones del robot en ROS.

        Esta función crea un mensaje JointState con la marca de tiempo actual,
        los nombres de las articulaciones y sus posiciones actuales.
        Luego, publica este mensaje en el tema ROS especificado.
        """
        
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.robot.nombres_articulaciones
        joint_state.position = self.robot.posiciones_articulaciones
        esperar_por_subscribers(self.pub_estados_articulaciones, 2)
        self.pub_estados_articulaciones.publish(joint_state)

    def crear_sliders(self):
        """
        Crea sliders para ajustar las posiciones de las articulaciones del robot.

        Return:
        Retorna una lista de objetos FloatSlider, cada uno correspondiente a una articulación del robot.

        Cada slider se configura con un rango de valores que corresponden a los límites de la articulación,
        un paso de ajuste, un valor inicial y una descripción.
        """
        
        sliders = []
        for i, name in enumerate(self.robot.nombres_articulaciones):
            
            slider = FloatSlider(min=self.robot.limites_inferiores[i], max=self.robot.limites_superiores[i], step=0.01, value=0.0, description=name, style={'description_width': 'initial'},
                                 layout=Layout(width='500px'))
            sliders.append(slider)
        return sliders

    def actualiza_posicion_articulaciones(self,cambio):
        """
        Actualiza las posiciones de las articulaciones del robot y publica los cambios.

        Parámetros de entrada:
        cambio: Objeto que contiene información sobre el cambio realizado en el deslizador.

        Esta función recorre todos los deslizadores, actualiza las posiciones de las articulaciones 
        del robot con los valores actuales de los deslizadores y luego publica el nuevo estado 
        de las articulaciones a través de ROS.
        """
        
        for i,slider in enumerate(self.sliders):
            self.robot.posiciones_articulaciones[i] = slider.value
        self.publicar_estados_articulaciones()

class Cinematica:
    def __init__(self, robot, frame_base, frame_final):
        """
        Inicializa una instancia para cálculo de cinemática del robot.

        Parámetros de entrada:
        robot: Objeto Robot conteniendo información del robot y URDF.
        frame_base: Nombre del frame base en la cadena cinemática.
        frame_final: Nombre del frame final en la cadena cinemática.

        Crea un árbol KDL a partir del modelo URDF, obtiene la cadena cinemática,
        y establece sovler para cinematica directa (CD), cinemática inversa (CI) y 
        cálculo de jacobianos.
        """
        
        self.robot = robot
        # Crear un árbol KDL desde URDF
        (ok, self.arbol_kdl) = treeFromUrdfModel(self.robot.robot_urdf)
        
        if not ok:
            raise Exception("Error al extraer árbol KDL del modelo URDF del robot")

        # Obtener la cadena
        self.frame_base = frame_base  
        self.frame_final = frame_final
        self.cadena = self.arbol_kdl.getChain(frame_base, frame_final)

        # Calculadores de FK y IK
        self.solver_cd = PyKDL.ChainFkSolverPos_recursive(self.cadena)
        self.solver_ci_vel = PyKDL.ChainIkSolverVel_pinv(self.cadena)
        # self.solver_ci = PyKDL.ChainIkSolverPos_NR_JL(self.cadena, self.robot.limites_inferioes_kdl_array, 
        #                                             self.robot.limites_superiores_kdl_array, self.solver_cd, 
        #                                             self.solver_ci_vel, 10000, 1e-5)
        self.solver_ci = IK(frame_base, frame_final, urdf_string=self.robot.descripcion_robot)
        self.solver_jacobiano = PyKDL.ChainJntToJacSolver(self.cadena)
    
    def wrap_to_pi(self, q):
        q = np.array(q, dtype=float)
        return (q + np.pi) % (2 * np.pi) - np.pi
    
    
    def calcular_cd(self, posiciones_articulaciones):
        """
        Calcula la cinemática directa (CD) dadas las posiciones de las articulaciones.

        Parámetros de entrada:
        posiciones_articulaciones: Lista de posiciones de las articulaciones.

        Return:
        Retorna un objeto Pose de ROS representando la pose del enlace final.
        """
        # Crear un array de articulaciones KDL a partir de las posiciones de las articulaciones
        posiciones_articulaciones_kdl = PyKDL.JntArray(len(posiciones_articulaciones))
        for i, posicion_articulacion in enumerate(posiciones_articulaciones):
            posiciones_articulaciones_kdl[i] = posicion_articulacion

        # Crear el frame que contendrá el resultado
        frame_kdl = PyKDL.Frame()

        # Calcular cinemática directa
        self.solver_cd.JntToCart(posiciones_articulaciones_kdl, frame_kdl)

        # Convertir frame KDL a pose ROS
        pose = frame_kdl_a_pose(frame_kdl)
        
        return pose
    
    def calcular_ci(self, posiciones_articulaciones_actuales, pose_deseada):
        """
        Calcula la cinemática inversa (CI) para alcanzar una pose deseada.

        Parámetros de entrada:
        posiciones_articulaciones_actuales (list): Posiciones de las articulaciones.
        pose_deseada (Pose): Pose deseada del efector final.

        Return:
        ok (bool): Indica si se encontró una solución válida.
        posiciones_articulaciones_resultado (list): Posiciones de las articulaciones para alcanzar la pose deseada.

        Convierte la pose deseada a un frame KDL, realiza el cálculo de la cinemática inversa y convierte los 
        resultados en una lista.
        """
        # Crea un array de articulaciones KDL a partir de las posiciones de las articulaciones    
        kdl_current_joint_values = PyKDL.JntArray(len(posiciones_articulaciones_actuales))
        for i, current_joint_value in enumerate(posiciones_articulaciones_actuales):
            kdl_current_joint_values[i] = current_joint_value
        # Crea un frame KDL a partir de la pose
        kdl_target_pose = PyKDL.Frame()
        kdl_target_pose.p = PyKDL.Vector(pose_deseada.position.x, pose_deseada.position.y, pose_deseada.position.z)
        kdl_target_pose.M = PyKDL.Rotation.Quaternion(pose_deseada.orientation.x, pose_deseada.orientation.y,
                                                      pose_deseada.orientation.z, pose_deseada.orientation.w)
        # Crea el array de articulaciones que contendrá el resultado
        kdl_result_joint_values = PyKDL.JntArray(len(posiciones_articulaciones_actuales))
        
        # Calcula la cinemática inversa
        # ik_valid = self.solver_ci.CartToJnt(kdl_current_joint_values, kdl_target_pose, kdl_result_joint_values)
        target_position_list = [pose_deseada.position.x, pose_deseada.position.y, pose_deseada.position.z] 
        target_orientation_list = [pose_deseada.orientation.x, pose_deseada.orientation.y,
                                   pose_deseada.orientation.z, pose_deseada.orientation.w]
        solution = self.solver_ci.get_ik(list(posiciones_articulaciones_actuales),
                                         target_position_list[0], target_position_list[1], target_position_list[2],
                                         target_orientation_list[0],target_orientation_list[1],target_orientation_list[2], target_orientation_list[3],
                                         0.0001, 0.0001, 0.0001, # Error admitido x,y,z
                                         0.002, 0.002, 0.002)# Error admitido rx, ry, rz
        
        result_joint_values = []
        
        if solution == None:
            ok = False
        else:
            ok = True
            # Conviere el resultado a una lista
            for i in range(len(solution)):
                result_joint_values.append(solution[i])
            result_joint_values = self.wrap_to_pi(result_joint_values)
                
        return ok,result_joint_values
    
    def calcular_jacobiana(self, posiciones_articulaciones_actuales):
        """ Calcula la jacobiana """
        # Create a KDL JointArray from the joint positions
        kdl_joint_positions = PyKDL.JntArray(len(posiciones_articulaciones_actuales))
        for i, joint_position in enumerate(posiciones_articulaciones_actuales):
            kdl_joint_positions[i] = joint_position
        # Create the Jacobian that will contain the result
        kdl_jacobian = PyKDL.Jacobian(len(posiciones_articulaciones_actuales))
        # Calculate the Jacobian
        self.solver_jacobiano.JntToJac(kdl_joint_positions, kdl_jacobian)
        # Convert the result to a numpy matrix
        jacobiana = np.zeros((kdl_jacobian.rows(), kdl_jacobian.columns()))
        for i in range(kdl_jacobian.rows()):
            for j in range(kdl_jacobian.columns()):
                jacobiana[i, j] = kdl_jacobian[i, j]
        return jacobiana
    
    def calcular_damped_jacobiana(self, jacobiana, damping_factor=0.01):
        # Compute the damped least squares of the Jacobian
        jacobian_transpose = np.transpose(jacobiana)
        damping_matrix = damping_factor * np.eye(jacobiana.shape[0])  # Square matrix with damping_factor on the diagonal
        damped_jacobiana = np.dot(jacobian_transpose, jacobiana) + damping_matrix
        return damped_jacobiana
    
    def calcular_pseudoinversa_jacobiana(self, jacobiana):
        # Compute the pseudo-inverse of the Jacobiana
        jacobiana_pinv = np.linalg.pinv(jacobiana)
        return jacobiana_pinv
        
    
class Kinematics:
    def __init__(self,robot, base_link, end_link):
        
        self.robot = robot
        # Create a KDL tree from URDF
        (ok, self.kdl_tree) = treeFromUrdfModel(self.robot.urdf_robot)
        
        if not ok:
            raise Exception("Failed to extract KDL tree from URDF robot model")

        # Get the chain
        self.base_link = base_link  
        self.end_link = end_link
        self.chain = self.kdl_tree.getChain(base_link, end_link)
        # Get the name of the joints in the chain
        
        self.min_joint_limits = PyKDL.JntArray(len(self.robot.joint_names))
        self.max_joint_limits = PyKDL.JntArray(len(self.robot.joint_names))  

        for joint in self.robot.joint_names:
            self.min_joint_limits[self.robot.joint_names.index(joint)] = self.robot.lower_limits[self.robot.joint_names.index(joint)]
            self.max_joint_limits[self.robot.joint_names.index(joint)] = self.robot.upper_limits[self.robot.joint_names.index(joint)]
        
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.ik_solver_vel = PyKDL.ChainIkSolverVel_pinv(self.chain)
        self.ik_solver = PyKDL.ChainIkSolverPos_NR_JL(self.chain, self.min_joint_limits, self.max_joint_limits, self.fk_solver, self.ik_solver_vel, 10000, 1e-5)
        self.jacobian_solver = PyKDL.ChainJntToJacSolver(self.chain)
        
    def compute_fk(self, joint_positions):
        # Create a KDL JointArray from the joint positions
        kdl_joint_positions = PyKDL.JntArray(len(joint_positions))
        for i, joint_position in enumerate(joint_positions):
            kdl_joint_positions[i] = joint_position
        # Create the frame that will contain the result
        kdl_frame = PyKDL.Frame()
        # Calculate forward kinematics
        self.fk_solver.JntToCart(kdl_joint_positions, kdl_frame)
        
        # Convert kdl frame to ros pose
        pose = frame_kdl_a_pose(kdl_frame)
        
        return pose
    
    def compute_ik(self, current_joint_values, target_pose):
        # Create a KDL JointArray from the joint positions
        kdl_current_joint_values = PyKDL.JntArray(len(current_joint_values))
        for i, current_joint_value in enumerate(current_joint_values):
            kdl_current_joint_values[i] = current_joint_value
        # Create a KDL Frame from the pose
        kdl_target_pose = PyKDL.Frame()
        kdl_target_pose.p = PyKDL.Vector(target_pose.position.x, target_pose.position.y, target_pose.position.z)
        kdl_target_pose.M = PyKDL.Rotation.Quaternion(target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w)
        # Create the joint array that will contain the result
        kdl_result_joint_values = PyKDL.JntArray(len(current_joint_values))
        # Calculate inverse kinematics
        ik_valid = self.ik_solver.CartToJnt(kdl_current_joint_values, kdl_target_pose, kdl_result_joint_values)
        if ik_valid < 0:
            ok = False
        else:
            ok = True
        # Convert the result to a list
        result_joint_values = []
        for i in range(kdl_result_joint_values.rows()):
            result_joint_values.append(kdl_result_joint_values[i])
        return ok,result_joint_values

    def compute_jacobian(self, joint_positions):
        # Create a KDL JointArray from the joint positions
        kdl_joint_positions = PyKDL.JntArray(len(joint_positions))
        for i, joint_position in enumerate(joint_positions):
            kdl_joint_positions[i] = joint_position
        # Create the Jacobian that will contain the result
        kdl_jacobian = PyKDL.Jacobian(len(joint_positions))
        # Calculate the Jacobian
        self.jacobian_solver.JntToJac(kdl_joint_positions, kdl_jacobian)
        # Convert the result to a numpy matrix
        jacobian = np.zeros((kdl_jacobian.rows(), kdl_jacobian.columns()))
        for i in range(kdl_jacobian.rows()):
            for j in range(kdl_jacobian.columns()):
                jacobian[i, j] = kdl_jacobian[i, j]
        return jacobian
    
    def compute_damped_jacobian(self, jacobian, damping_factor=0.01):
        # Compute the damped least squares of the Jacobian
        jacobian_transpose = np.transpose(jacobian)
        damping_matrix = damping_factor * np.eye(jacobian.shape[0])  # Square matrix with damping_factor on the diagonal
        damped_jacobian = np.dot(jacobian_transpose, jacobian) + damping_matrix
        return damped_jacobian
    
    def compute_pseudo_inverse_jacobian(self, jacobian):
        # Compute the pseudo-inverse of the Jacobian
        jacobian_pinv = np.linalg.pinv(jacobian)
        return jacobian_pinv

def create_trajectory(start_pose, end_pose, num_points):
    # Create a list of waypoints
    waypoints = []
    for i in range(num_points):
        # Compute the interpolation parameter
        s = float(i) / float(num_points - 1)
        # Interpolate position
        px = (1 - s) * start_pose.position.x + end_pose.position.x * s
        py = (1 - s) * start_pose.position.y + end_pose.position.y * s
        pz = (1 - s) * start_pose.position.z + end_pose.position.z * s
        # Interpolate orientation
        # Perform SLERP
        # Define start and end quaternions
        start_quaternion = [start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w]
        end_quaternion = [end_pose.orientation.x, end_pose.orientation.y, end_pose.orientation.z, end_pose.orientation.w]
        interpolated_quaternion = tf_trans.quaternion_slerp(start_quaternion, end_quaternion, s)

        # Create a Pose message
        pose = Pose()
        pose.position.x = px
        pose.position.y = py
        pose.position.z = pz
        pose.orientation.x = interpolated_quaternion[0]
        pose.orientation.y = interpolated_quaternion[1]
        pose.orientation.z = interpolated_quaternion[2]
        pose.orientation.w = interpolated_quaternion[3]
        # Add the pose to the list of waypoints
        waypoints.append(pose)
    return waypoints

def get_pose_error(current_pose, target_pose):
    position_error = [target_pose.position.x - current_pose.position.x, target_pose.position.y - current_pose.position.y, target_pose.position.z - current_pose.position.z]
    
        # Convert quaternions to PyKDL rotations
    current_rotation = PyKDL.Rotation.Quaternion(current_pose.orientation.x,
                                                 current_pose.orientation.y,
                                                 current_pose.orientation.z,
                                                 current_pose.orientation.w)
    target_rotation = PyKDL.Rotation.Quaternion(target_pose.orientation.x,
                                                target_pose.orientation.y,
                                                target_pose.orientation.z,
                                                target_pose.orientation.w)

    # Compute the relative rotation from current to target
    relative_rotation = current_rotation.Inverse() * target_rotation

    # Convert the relative rotation to axis-angle representation
    angle, axis = relative_rotation.GetRotAngle()
    orientation_error = [angle * axis[0], angle * axis[1], angle * axis[2]]

    # Combine position and orientation errors
    error = np.zeros(6)  # 6D vector for 3D position and 3D orientation
    error[0:3] = position_error
    error[3:6] = orientation_error
    return error
    
    


def quaternion_to_rotation_matrix(quat):
    # Convert a Quaternion to a PyKDL Rotation
    return PyKDL.Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)

def compute_local_axis_orientations(quaternion):

    
    # Convert the orientation quaternion to a PyKDL rotation
    rotation = PyKDL.Rotation.Quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)

    # Create rotations for each local axis
    rot_x = PyKDL.Rotation.RotX(1.0) * rotation
    rot_y = PyKDL.Rotation.RotY(1.0) * rotation
    rot_z = PyKDL.Rotation.RotZ(1.0) * rotation

    # Convert rotations to quaternions
    quat_x = rot_x.GetQuaternion()
    quat_y = rot_y.GetQuaternion()
    quat_z = rot_z.GetQuaternion()

    return quat_x, quat_y, quat_z


class ConfiguradorSlidersEfectorFinal():
    
    def __init__(self, robot=None, pub_estados_articulaciones=None, pub_marcador_efector_final=None, cinematica=None):
        """
        Inicializa un configurador de sliders para ajustar la pose del efector final.

        Parámetros de entrada:
        robot (Robot): Instancia de la clase Robot.
        pub_estados_articulaciones (rospy.Publisher): Publicador ROS para los estados de las articulaciones.
        pub_marcador_efector_final (rospy.Publisher): Publicador ROS para el marcador del efector final.
        cinematica (Cinematica): Instancia de la clase Cinematica para cálculos FK e IK.

        Crea sliders para ajustar la pose del efector final en términos de posición (x, y, z) y orientación (roll, pitch, yaw).
        """
        self.robot = robot
        self.pub_estados_articulaciones = pub_estados_articulaciones
        self.pub_marcador_efector_final = pub_marcador_efector_final
        self.cinematica = cinematica
        self.pose_objetivo = Pose()
        self.nombres_sliders = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        self.limites_minimos_sliders = [-1.5, -1.5, 0.0, -3.14, -3.14, -3.14]
        self.limites_maximos_sliders = [1.5, 1.5, 1.5, 3.14, 3.14, 3.14]
        self.sliders = self.crear_sliders()
        vbox = VBox([*self.sliders])
        display(vbox)
        for slider in self.sliders:
            slider.observe(self.actualizar_pose_efector_final, names='value')
            
    def crear_sliders(self):
        """
        Crea sliders para ajustar los componentes de la pose del efector final.

        Return:
        Retorna una lista de objetos FloatSlider, cada uno para un componente diferente de la pose (x, y, z, roll, pitch, yaw).

        Cada slider se configura con un rango de valores, un paso de ajuste, un valor inicial y una descripción.
        """
        sliders = []
        for i, nombre in enumerate(self.nombres_sliders):
            slider = FloatSlider(min=self.limites_minimos_sliders[i], max=self.limites_maximos_sliders[i], step=0.01, value=0.0, description=nombre, style={'description_width': 'initial'},
                                    layout=Layout(width='500px'))
            sliders.append(slider)
        return sliders
    
    def actualizar_pose_efector_final(self, cambio):
        """
        Actualiza la pose del efector final basada en los valores de los sliders.

        Parámetros de entrada:
        cambio: Información sobre el cambio realizado en el deslizador (no utilizado directamente).

        Esta función recoge los valores actuales de los sliders, los convierte en una pose
        (posición y orientación), y luego publica esta pose como un marcador en RViz.
        """
        
        x = self.sliders[0].value
        y = self.sliders[1].value
        z = self.sliders[2].value
        roll = self.sliders[3].value
        pitch = self.sliders[4].value
        yaw = self.sliders[5].value
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        q = PyKDL.Rotation.RPY(roll, pitch, yaw).GetQuaternion()
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        self.pose_objetivo = pose
        esperar_por_subscribers(self.pub_marcador_efector_final, 2)
        publicar_marcador(pub_marcador=self.pub_marcador_efector_final, pose_efector_final=pose)
    
    def obtener_pose_sliders(self):
        """
        Devuelve la pose objetivo actual del efector final.

        Return:
        Retorna un objeto Pose que representa la pose objetivo actual del efector final.

        Esta función simplemente devuelve la pose objetivo que se ha configurado mediante los deslizadores.
        """
        
        return self.pose_objetivo

class Obstaculo:
    def __init__(self, tipo, pose, dimensiones, nombre):
        """
        Inicializa un obstáculo.

        Parámetros de entrada:
        tipo (str): Tipo de obstáculo (cubo, esfera, cilindro).
        pose (Pose): Pose del obstáculo [x, y, z, qx, qy, qz, qw]
        dimensiones (list): Lista de dimensiones del obstáculo (x, y, z).
        si esfera solo hacemos caso a la primera dimension
        si cilindro solo hacemos caso a la primera y segunda dimension (radio y altura)
        si cubo hacemos caso a las tres dimensiones (lado_x, lado_y y lado_z)
        Crea un objeto para el obstáculo.
        """
        
        self.tipo = tipo
        self.pose = pose
        self.dimensiones = dimensiones
        self.nombre = nombre
        
class DetectorColisionesGripperFlotante:
    def __init__(self, gripper_name, obstaculos=[]):
        """
        Inicializa el detector de colisiones enfocado en el gripper y obstáculos específicos.

        Parámetros:
        - gripper_dimensions (list): Dimensiones del gripper [x, y, z].
        - obstaculos (list): Lista de obstáculos, donde cada obstáculo es una instancia de la clase Obstaculo.
        """
        self.gripper_name = gripper_name
        self.obstaculos = obstaculos

        rospack = rospkg.RosPack()
        package_path = rospack.get_path('manipulacion_pkg')
        # Inicializar el modelo de colisiones
        print(package_path+'/urdf/'+self.gripper_name+'.urdf')
        self.modelo = pin.buildModelFromUrdf(package_path+'/urdf/'+self.gripper_name+'.urdf',pin.JointModelFreeFlyer())
        # Load collision geometries
        self.modelo_geom = pin.buildGeomFromUrdf(self.modelo, package_path+'/urdf/'+self.gripper_name+'.urdf', pin.GeometryType.COLLISION)


        
        for i,obstaculo in enumerate(obstaculos):
            
            geom_obj = self.crear_objeto_geom(obstaculo)
            if geom_obj:
                self.modelo_geom.addGeometryObject(geom_obj)
                
                joint_id = self.modelo.addJoint(
                            0,
                            pin.JointModelFreeFlyer(),
                            pin.SE3.Identity(),
                            joint_name='joint_'+str(i),
                            max_effort=1000 * np.ones(6),
                            max_velocity=1000 * np.ones(6),
                            min_config=np.array([-1, -1, -1, 0., 0., 0., 1.]),
                            max_config=np.array([1, 1, 1, 0., 0., 0., 1.]))

                com = np.array([0, 0, 0])  
                moment_inertia = np.diag([
                        0.0001,
                        0.0001,
                        0.0001,
                ]) 

                self.modelo.appendBodyToJoint(
                joint_id,
                pin.Inertia(0, com, moment_inertia),
                pin.SE3.Identity()
                )
        self.q = pin.neutral(self.modelo)
        self.modelo_geom.addAllCollisionPairs()
        self.data = self.modelo.createData()
        self.geom_data = pin.GeometryData(self.modelo_geom)

    def crear_objeto_geom(self, obstaculo):
        """
        Crea un objeto GeometryObject basado en la información del obstáculo.
        """
        if obstaculo.tipo == 'cubo':
            pose = obstaculo.pose
            posicion = np.array(pose[:3])
            orientacion = pin.Quaternion(pose[3], pose[4], pose[5], pose[6]).toRotationMatrix()
            # Crear el objeto SE3 para la pose del obstáculo
            se3_pose = pin.SE3(orientacion, posicion)
            
            return pin.GeometryObject(
                obstaculo.nombre,
                0,  # Se asocia con el cuerpo base
                hppfcl.Box(obstaculo.dimensiones[0],obstaculo.dimensiones[1], obstaculo.dimensiones[2]),
                se3_pose  
                )
        # Añadir otros tipos de obstáculos si es necesario
        return None

    def hay_colision(self, pose_gripper):
        """
        Verifica si hay colisión entre el gripper y los obstáculos configurados.

        Parámetros:
        - pose_gripper (list): Pose del gripper [x, y, z, qx, qy, qz, qw].
        """
        
        # Convertir la orientación de cuaternión a una matriz de rotación
        rotation_matrix = pin.Quaternion(pose_gripper[3], pose_gripper[4], pose_gripper[5], pose_gripper[6]).toRotationMatrix()

      # Crear el objeto SE3 con la matriz de rotación y la posición
        transformation = pin.SE3(rotation_matrix, np.array(pose_gripper[:3]))

        # Configurar la pose del gripper
        self.data.oMf[0] = transformation
        
        # Detectar colisiones
        q = pin.SE3ToXYZQUAT(transformation)  # Convierte SE3 a un vector compatible con Pinocchio
        
        for i in range(len(self.obstaculos)):
          # new_values = np.array([self.obstaculos[i].pose[0], self.obstaculos[i].pose[1], self.obstaculos[i].pose[2],
                              #  self.obstaculos[i].pose[3], self.obstaculos[i].pose[4], self.obstaculos[i].pose[5], self.obstaculos[i].pose[6]])
          new_values = np.array([0,0,0, 1,0,0,0])
          q = np.concatenate((q, new_values))
        
        
        # Luego, intenta llamar a computeCollisions con q
        pin.computeCollisions(self.modelo, self.data, self.modelo_geom, self.geom_data, q, True)
        for i in range(len(self.modelo_geom.collisionPairs)):
            cr = self.geom_data.collisionResults[i]
            if cr.isCollision():
                return True
        return False

class DetectorColisiones:
    def __init__(self, usa_brazo_robotico=True, usa_gripper=False, gripper_dimensions = [], obstaculos=[]):
        self.usa_brazo_robotico = usa_brazo_robotico
        self.usa_gripper = usa_gripper
        self.obstaculos = obstaculos
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('manipulacion_pkg')

        self.modelo = None
        self.modelo_geom = None
        self.last_joint_id = None
        
        if self.usa_gripper:
            if gripper_dimensions == []:
                raise Exception("No se han especificado las dimensiones del gripper")
            self.gripper_dimensions = gripper_dimensions
        
        if usa_brazo_robotico:
            self.modelo = pin.buildModelFromUrdf(package_path+'/urdf/'+'ur10.urdf',pin.JointModelFreeFlyer())
            # Load collision geometries
            self.modelo_geom = pin.buildGeomFromUrdf(self.modelo,package_path+'/urdf/'+'ur10.urdf', pin.GeometryType.COLLISION)

        
            if self.usa_gripper:
                print("usando gripper")
                # Create a gripper collision box
            
                self.last_joint_id = self.modelo.njoints - 1

                gripper_obstacle_box = pin.GeometryObject(
                'gripper',  
                self.last_joint_id,  # Attach to the last joint
                hppfcl.Box(self.gripper_dimensions[0], self.gripper_dimensions[1], self.gripper_dimensions[2]),
                pin.SE3(pin.Quaternion(0.0, 0, 0, 1.0).matrix(), np.array([0.0, 0, self.gripper_dimensions[2]/2.0]))  # Initial placement relative to the joint
                )

                self.modelo_geom.addGeometryObject(gripper_obstacle_box)
                joint_id_gripper = self.modelo.addJoint(
                    self.last_joint_id,
                    pin.JointModelFreeFlyer(),
                    pin.SE3.Identity(),
                    joint_name='joint_gripper',
                    max_effort=1000 * np.ones(6),
                    max_velocity=1000 * np.ones(6),
                    min_config=np.array([-1, -1, -1, 0., 0., 0., 1.]),
                    max_config=np.array([1, 1, 1, 0., 0., 0., 1.]))
                M_cyl = 3.
                com = np.array([0, 0, 0])  
                moment_inertia = np.diag([
                        0.0001,
                        0.0001,
                        0.0001,
                ]) 
                self.modelo.appendBodyToJoint(
                    joint_id_gripper,
                    pin.Inertia(M_cyl, com, moment_inertia),
                    pin.SE3.Identity()
                )


        
        
        # Añadimos un objeto al modelo de colisiones para el suelo
        
        suelo_geom_col = pin.GeometryObject(
                                'suelo',
                                0,
                                hppfcl.Box(4.0, 4.0, 0.1),
                                pin.SE3.Identity())


        self.modelo_geom.addGeometryObject(suelo_geom_col)
        joint_id = self.modelo.addJoint(
            0,
            pin.JointModelFreeFlyer(),
            pin.SE3.Identity(),
            joint_name='joint_suelo',
            max_effort=1000 * np.ones(6),
            max_velocity=1000 * np.ones(6),
            min_config=np.array([-1, -1, -1, 0., 0., 0., 1.]),
            max_config=np.array([1, 1, 1, 0., 0., 0., 1.]))

        com = np.array([0, 0, 0])  
        moment_inertia = np.diag([
                0.0001,
                0.0001,
                0.0001,
        ]) 

        self.modelo.appendBodyToJoint(
        joint_id,
        pin.Inertia(0, com, moment_inertia),
        pin.SE3.Identity()
        )
        
        
        #TODO: Añadir obstaculos al modelo de colisiones
        for obstaculo in self.obstaculos:
          if obstaculo.tipo == 'cubo':
            print("obstaculo")
            geom_obj = self.crear_objeto_geom(obstaculo)
            if geom_obj:
              self.modelo_geom.addGeometryObject(geom_obj)
              joint_id_obs = self.modelo.addJoint(
                          0,
                          pin.JointModelFreeFlyer(),
                          pin.SE3.Identity(),
                          joint_name='joint_'+obstaculo.nombre,
                          max_effort=1000 * np.ones(6),
                          max_velocity=1000 * np.ones(6),
                          min_config=np.array([-1, -1, -1, 0., 0., 0., 1.]),
                          max_config=np.array([1, 1, 1, 0., 0., 0., 1.]))

              com = np.array([0, 0, 0])  
              moment_inertia = np.diag([
                      0.0001,
                      0.0001,
                      0.0001,
              ]) 

              self.modelo.appendBodyToJoint(
              joint_id_obs,
              pin.Inertia(0, com, moment_inertia),
              pin.SE3.Identity()
              )
            
        
        self.q = pin.neutral(self.modelo)
        print(self.q)
        self.modelo_geom.addAllCollisionPairs()

        print("num collision pairs - initial:",len(self.modelo_geom.collisionPairs))

        print("base_inertia: ", self.modelo_geom.getGeometryId('base_link_inertia_0'))
        print("shoulder: ", self.modelo_geom.getGeometryId('shoulder_link_0'))
        print("upper: ", self.modelo_geom.getGeometryId('upper_arm_link_0'))
        print("forearm: ", self.modelo_geom.getGeometryId('forearm_link_0'))
        print("wrist1: ", self.modelo_geom.getGeometryId('wrist_1_link_0'))
        print("wrist2: ", self.modelo_geom.getGeometryId('wrist_2_link_0'))
        print("wrist3: ", self.modelo_geom.getGeometryId('wrist_3_link_0'))
        print("gripper: ", self.modelo_geom.getGeometryId('gripper'))
        print("suelo: ", self.modelo_geom.getGeometryId('suelo'))
        for obstaculo in self.obstaculos:
          print(obstaculo.nombre, ": ", self.modelo_geom.getGeometryId(obstaculo.nombre))
        
        for collision_pair in self.modelo_geom.collisionPairs:
        #     print(collision_pair)
            if self.modelo_geom.getGeometryId('suelo') == collision_pair.second and self.modelo_geom.getGeometryId('base_link_0') == collision_pair.first:
                self.modelo_geom.removeCollisionPair(collision_pair)
            if self.modelo_geom.getGeometryId('shoulder_link_0') == collision_pair.first and self.modelo_geom.getGeometryId('upper_arm_link_0') == collision_pair.second:
                self.modelo_geom.removeCollisionPair(collision_pair)
            if self.modelo_geom.getGeometryId('upper_arm_link_0') == collision_pair.first and self.modelo_geom.getGeometryId('forearm_link_0') == collision_pair.second:
                self.modelo_geom.removeCollisionPair(collision_pair)
            if self.modelo_geom.getGeometryId('forearm_link_0') == collision_pair.first and self.modelo_geom.getGeometryId('wrist_1_link_0') == collision_pair.second:
                self.modelo_geom.removeCollisionPair(collision_pair)
            if self.modelo_geom.getGeometryId('wrist_1_link_0') == collision_pair.first and self.modelo_geom.getGeometryId('wrist_2_link_0') == collision_pair.second:
                self.modelo_geom.removeCollisionPair(collision_pair)
            if self.modelo_geom.getGeometryId('wrist_2_link_0') == collision_pair.first and self.modelo_geom.getGeometryId('wrist_3_link_0') == collision_pair.second:
                self.modelo_geom.removeCollisionPair(collision_pair)
            if self.modelo_geom.getGeometryId('wrist_3_link_0') == collision_pair.first and self.modelo_geom.getGeometryId('gripper') == collision_pair.second:
                self.modelo_geom.removeCollisionPair(collision_pair)

            #     self.modelo_geom.removeCollisionPair(collision_pair)
            # if self.modelo_geom.getGeometryId('world/hori_surface') == collision_pair.first and self.modelo_geom.getGeometryId('world/obstacle_box') == collision_pair.second:
            #     self.modelo_geom.removeCollisionPair(collision_pair)

        self.data = self.modelo.createData()
        self.geom_data = pin.GeometryData(self.modelo_geom)
    
    def hay_colision(self, configuracion_articulaciones):
        
        # Verificar colisiones dado un conjunto de configuraciones de articulaciones
        # 'configuracion_articulaciones' puede ser la pose del gripper si es_gripper_flotante es True
        if self.usa_brazo_robotico:
            self.q[7:13] = configuracion_articulaciones
        # print(self.q)
        pin.computeCollisions(self.modelo, self.data, self.modelo_geom, self.geom_data, self.q, True)
        for i in range(len(self.modelo_geom.collisionPairs)):
            cr = self.geom_data.collisionResults[i]
            # print(self.modelo_geom.collisionPairs[i])
            # print(cr.isCollision())
            if cr.isCollision():
                return True
        return False
    
    def crear_objeto_geom(self, obstaculo):
        """
        Crea un objeto GeometryObject basado en la información del obstáculo.
        """
        if obstaculo.tipo == 'cubo':
            pose = obstaculo.pose
            posicion = np.array(pose[:3])
            orientacion = pin.Quaternion(pose[3], pose[4], pose[5], pose[6]).toRotationMatrix()
            # Crear el objeto SE3 para la pose del obstáculo
            se3_pose = pin.SE3(orientacion, posicion)
            return pin.GeometryObject(
                obstaculo.nombre,
                0,  # Se asocia con el cuerpo base
                hppfcl.Box(obstaculo.dimensiones[0],obstaculo.dimensiones[1], obstaculo.dimensiones[2]),
                se3_pose  # Posición inicial neutra
                )
        # Añadir otros tipos de obstáculos si es necesario
        return None


class SimulacionGripperFlotante():
    def __init__(self, nombre_gripper_gazebo):
        self.pose_gripper = None
        self.pose_objeto = None
        self.nombre_gripper_gazebo = nombre_gripper_gazebo
        self.tipo_gripper = rospy.get_param('/tipo_gripper')
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.nombre_articulaciones = []
        self.pub_posicion_articulaciones_command = None
        self.posicion_articulaciones_gripper_abierto = []
        
    def obtener_pose_gripper(self):
        resp = self.get_state(self.nombre_gripper_gazebo, 'world')
        f_gripper_world = PyKDL.Frame()
        f_gripper_world.p = PyKDL.Vector(resp.pose.position.x, resp.pose.position.y, resp.pose.position.z)
        f_gripper_world.M = PyKDL.Rotation.Quaternion(resp.pose.orientation.x, resp.pose.orientation.y, resp.pose.orientation.z, resp.pose.orientation.w)
        
        self.pose_gripper = f_gripper_world
        return self.pose_gripper
    
    def fijar_pose_gripper(self, pose_gripper_world):
        state = ModelState()
        #pose_gripper_world is a PyKDL.Frame
        state.model_name = self.nombre_gripper_gazebo
        state.pose.position.x = pose_gripper_world.p[0]
        state.pose.position.y = pose_gripper_world.p[1]
        state.pose.position.z = pose_gripper_world.p[2]
        state.pose.orientation.x = pose_gripper_world.M.GetQuaternion()[0]
        state.pose.orientation.y = pose_gripper_world.M.GetQuaternion()[1]
        state.pose.orientation.z= pose_gripper_world.M.GetQuaternion()[2]
        state.pose.orientation.w = pose_gripper_world.M.GetQuaternion()[3]

        resp = self.set_state(state)
        return resp.success
    
    def obtener_pose_objeto(self, nombre_objeto_gazebo):
        resp = self.get_state(nombre_objeto_gazebo, 'world')
        f_objeto_world = PyKDL.Frame()
        f_objeto_world.p = PyKDL.Vector(resp.pose.position.x, resp.pose.position.y, resp.pose.position.z)
        f_objeto_world.M = PyKDL.Rotation.Quaternion(resp.pose.orientation.x, resp.pose.orientation.y, resp.pose.orientation.z, resp.pose.orientation.w)
        
        self.pose_objeto = f_objeto_world
        return self.pose_objeto
    
    def configurar_gripper(self):

      # Construir la ruta al archivo de configuración basado en el nombre del gripper
      rospack = rospkg.RosPack()
      configuracion_gripper = rospack.get_path('manipulacion_pkg') + '/config/grippers/' + self.tipo_gripper + '_hand_config.yaml'

      # Cargar el archivo de configuracion
      with open(configuracion_gripper, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
      joint_positions_open =config['joint_positions_open'] 
      self.posicion_articulaciones_gripper_abierto = list(joint_positions_open.values())
      topic_pub = config['topic_pub']
      self.nombre_articulaciones = config['joint_names']
      self.pub_posicion_articulaciones_command = rospy.Publisher(topic_pub, JointTrajectory, queue_size=10)
      
    def set_posicion_articulaciones(self, posicion_articulaciones):    
      trajectory = JointTrajectory()
      trajectory.header.stamp = rospy.Time.now()
      trajectory.joint_names = self.nombre_articulaciones
      point = JointTrajectoryPoint()
      print("Gripper posicion articulaciones: ", posicion_articulaciones)
      point.positions = posicion_articulaciones
      point.time_from_start = rospy.Duration(2)
      trajectory.points = [point]
      esperar_por_subscribers(self.pub_posicion_articulaciones_command, 2)
      self.pub_posicion_articulaciones_command.publish(trajectory)
      rospy.sleep(2)
      
    def abrir_gripper(self):
      self.set_posicion_articulaciones(self.posicion_articulaciones_gripper_abierto)
    
    def get_tipo_gripper(self):
      return self.tipo_gripper
    

class SimulacionGripper():
    def __init__(self, nombre_gripper_gazebo):
        self.pose_objeto = None
        self.nombre_gripper_gazebo = nombre_gripper_gazebo
        self.tipo_gripper = rospy.get_param('/tipo_gripper')
        rospy.wait_for_service('/gazebo/get_model_state')
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.nombre_articulaciones_gripper = []
        self.pub_posicion_articulaciones_gripper_command = None
        self.posicion_articulaciones_gripper_abierto = []

    
    def obtener_pose_objeto(self, nombre_objeto_gazebo):
        resp = self.get_state(nombre_objeto_gazebo, 'world')
        f_objeto_world = PyKDL.Frame()
        f_objeto_world.p = PyKDL.Vector(resp.pose.position.x, resp.pose.position.y, resp.pose.position.z)
        f_objeto_world.M = PyKDL.Rotation.Quaternion(resp.pose.orientation.x, resp.pose.orientation.y, resp.pose.orientation.z, resp.pose.orientation.w)
        
        self.pose_objeto = f_objeto_world
        return self.pose_objeto
    
    def configurar_gripper(self):

      # Construir la ruta al archivo de configuración basado en el nombre del gripper
      rospack = rospkg.RosPack()
      configuracion_gripper = rospack.get_path('manipulacion_pkg') + '/config/grippers/' + self.tipo_gripper + '_hand_config.yaml'

      # Cargar el archivo de configuracion
      with open(configuracion_gripper, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
      joint_positions_open =config['joint_positions_open'] 
      self.posicion_articulaciones_gripper_abierto = list(joint_positions_open.values())
      topic_pub = config['topic_pub']
      self.nombre_articulaciones_gripper = config['joint_names']
      self.pub_posicion_articulaciones_gripper_command = rospy.Publisher(topic_pub, JointTrajectory, queue_size=10)
      
    def set_posicion_articulaciones(self, posicion_articulaciones):    
      trajectory = JointTrajectory()
      trajectory.header.stamp = rospy.Time.now()
      trajectory.joint_names = self.nombre_articulaciones_gripper
      point = JointTrajectoryPoint()
      print("Gripper posicion articulaciones: ", posicion_articulaciones)
      point.positions = posicion_articulaciones
      point.time_from_start = rospy.Duration(2)
      trajectory.points = [point]
      esperar_por_subscribers(self.pub_posicion_articulaciones_gripper_command, 2)
      self.pub_posicion_articulaciones_gripper_command.publish(trajectory)
      rospy.sleep(2)
      

      
    def abrir_gripper(self):
      self.set_posicion_articulaciones(self.posicion_articulaciones_gripper_abierto)
    
    def get_tipo_gripper(self):
      return self.tipo_gripper





class NodeJointSpace:
    def __init__(self, joint_angles):
        self.joint_angles = joint_angles
        self.parent = None
        self.cost = float('inf')
      

class BiRRTJointSpace:
    def __init__(self, start, goal, joint_limits, expand_dis=0.1, max_iter=500, search_radius=1.5, collision_detector=None):
        self.start = NodeJointSpace(start)
        self.start.cost = 0
        self.goal = NodeJointSpace(goal)
        self.joint_limits = joint_limits
        self.expand_dis = expand_dis
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.detector_colisiones = collision_detector
        self.node_list = [self.start]
        self.connect_threshold = self.search_radius
    def plan(self):
        start_root = self.start
        goal_root = self.goal
        
        tree_start = [start_root]
        tree_goal = [goal_root]
        
        self.node_list = tree_start
        
        for i in range(self.max_iter):

            if i % 2 == 0:
                tree_a, root_a = tree_start, start_root
                tree_b, root_b = tree_goal, goal_root
            else:
                tree_a, root_a = tree_goal, goal_root
                tree_b, root_b = tree_start, start_root


            target_node = root_b
            rnd_node = self.get_random_node(target_node)
            

            nearest_node = self.get_nearest_node(rnd_node, tree_a)

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, nearest_node):
                continue

            tree_a.append(new_node)
            

            nearest_other = self.get_nearest_node(new_node, tree_b)
            
            connect_node = self.steer(nearest_other, new_node, self.expand_dis)

            if self.check_collision(connect_node, nearest_other):
                continue

            if self.get_distance(connect_node, new_node) > self.connect_threshold:
                continue

            tree_b.append(connect_node)


            if tree_a is tree_start:
                path_start = self.build_path_to_root(new_node)         # start_root → new_node
                path_goal = self.build_path_to_root(connect_node)      # goal_root → connect_node
            else:
                path_start = self.build_path_to_root(connect_node)     # start_root → connect_node
                path_goal = self.build_path_to_root(new_node)          # goal_root → new_node

            path_goal.reverse()
            

            full_path = path_start + path_goal[1:]

            return full_path

        return None


    def get_random_node(self, target_node):
        """
        Generates a random node within the joint limits or the target node.
        Args:
            target_node (NodeJointSpace): The target node (goal or start) to sample occasionally.
        Returns:
            NodeJointSpace: The randomly generated node or the target node.
        """
        # Define the probability of sampling the target node
        target_sample_rate = 0.2  # e.g., 10% chance to sample the target node

        if np.random.rand() < target_sample_rate:
            return target_node
        else:
            joint_angles = [np.random.uniform(low, high) for (low, high) in self.joint_limits]
            return NodeJointSpace(joint_angles)

    def get_nearest_node(self, node, node_list):
        distances = [self.get_distance(n, node) for n in node_list]
        nearest_idx = np.argmin(distances)
        return node_list[nearest_idx]

      
    def steer(self, from_node, to_node, extend_length=float('inf')):
        direction = np.array(to_node.joint_angles) - np.array(from_node.joint_angles)
        distance = np.linalg.norm(direction)

        if distance > extend_length:
            direction = direction / distance
            new_joint_angles = np.array(from_node.joint_angles) + extend_length * direction
            new_node = NodeJointSpace(new_joint_angles.tolist())
            new_node.parent = from_node
        else:
          new_joint_angles = np.array(to_node.joint_angles)
        # Ensure new_joint_angles are within joint limits
        for i, (low, high) in enumerate(self.joint_limits):
            new_joint_angles[i] = np.clip(new_joint_angles[i], low, high)

        new_node = NodeJointSpace(new_joint_angles.tolist())
        new_node.parent = from_node
        return new_node

    def build_path_to_root(self,node):
        """Devuelve la trayectoria desde la raíz de ese árbol hasta node."""
        path = []
        current = node
        while current is not None:
            path.append(current.joint_angles)
            current = current.parent
        path.reverse()
        return path



    def check_collision(self, to_node, from_node):
        """
        Checks if the path between two nodes collides with any obstacles.
        Args:
            to_node, from_node (NodeJointSpace): Nodes to check the path between.
        Returns:
            bool: True if there is a collision, False otherwise.
        """

        # if DetectorColisiones.hay_colision(self, to_node.joint_angles)
        # Number of steps for interpolation
        num_steps = 20  # Adjust this based on desired granularity

        # # Linear interpolation
        for step in range(1, num_steps + 1):
            ratio = step / float(num_steps)
            interpolated_joints = (1 - ratio) * np.array(from_node.joint_angles) + ratio * np.array(to_node.joint_angles)
            
            # Check for collision at the interpolated position
            if self.detector_colisiones.hay_colision(interpolated_joints):
                return True  # Collision detected
        if self.detector_colisiones.hay_colision(to_node.joint_angles):
            return True
        if self.detector_colisiones.hay_colision(from_node.joint_angles):
            return True
        return False  # No collision detect

    def find_near_nodes(self, new_node):
        n = len(self.node_list) + 1
        r = min(self.search_radius, (np.log(n) / n)**(1/len(self.joint_limits)))
        distances = [self.get_distance(node, new_node) for node in self.node_list]
        near_nodes = [self.node_list[idx] for idx, d in enumerate(distances) if d < r]
        return near_nodes

    def choose_parent(self, near_nodes, new_node):
        if not near_nodes:
            return None
        costs = []
        for node in near_nodes:
            if not self.check_collision(new_node, node):
                costs.append(node.cost + self.get_distance(node, new_node))
            else:
                costs.append(float('inf'))
        min_cost = min(costs)
        if min_cost == float('inf'):
            return None
        min_cost_idx = costs.index(min_cost)
        new_node.cost = min_cost
        new_node.parent = near_nodes[min_cost_idx]
        return new_node

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            no_collision = not self.check_collision(node, new_node)
            improved_cost = new_node.cost + self.get_distance(new_node, node) < node.cost
            if no_collision and improved_cost:
                node.parent = new_node
                node.cost = new_node.cost + self.get_distance(new_node, node)

    def get_distance(self, node1, node2):
        return np.linalg.norm(np.subtract(node1.joint_angles, node2.joint_angles))
      
    def check_goal_path(self, node):
        # Attempt to directly connect node to goal
        direct_to_goal = self.steer(node, self.goal)
        
        if not self.check_collision(direct_to_goal, node):
            # Direct path is feasible, update goal node's parent and cost if this path is better
            direct_cost = node.cost + self.get_distance(node, self.goal)
            if direct_cost < self.goal.cost or self.goal.parent is None:
                self.goal.parent = node
                self.goal.cost = direct_cost
                return True
        return False


    def generate_final_path(self, goal_node):
      """
      Generates the final path from start to goal by tracing back from the goal node.
      Args:
          goal_node (NodeJointSpace): The goal node from which to trace back to the start.
      Returns:
          list: The sequence of joint angles from start to goal.
      """
      path = []
      current_node = goal_node
      while current_node.parent is not None:
          path.append(current_node.joint_angles)
          current_node = current_node.parent
      path.append(self.start.joint_angles)  # Don't forget to add the start node at the end
      path.reverse()  # Reverse the path to start from the beginning
      return path
