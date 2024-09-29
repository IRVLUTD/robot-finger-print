#External Libraries
import numpy as np
from tqdm import tqdm
import os
import argparse
import sys
import time

def make_parser():
    """ Input Parser """
    parser = argparse.ArgumentParser(description='Standalone script for grasp filtering.')
    parser.add_argument('--headless', type=bool, help='Running Program in headless mode',
                        default=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--gripper_dir', type=str, help='Directory of Gripper urdf/usd', default='')
    parser.add_argument('--output_dir', type=str, help='Output directory for results', default='')
    parser.add_argument('--device', type=int, help='Gpu to use', default=0)
    parser.add_argument('--test_time', type=int, help='Total time for each grasp test', default=5)
    parser.add_argument('--num_w', type=int, help='Number of spheres to test', default=10)
    parser.add_argument('--step_size', type=float, help='Radius step size', default=0.005)
    parser.add_argument('--init_r', type=float, help='Initial Radius', default=0.03)
    parser.add_argument('--gripper_ID', type=str, help='gripper name to test')
    parser.add_argument('--controller', type=str,
                        help='Gripper Controller to use while testing, should match the controller dictionary in the Manager Class',
                        default='transfer_default')
    parser.add_argument('--/log/level', type=str, help='isaac sim logging arguments', default='', required=False)
    parser.add_argument('--/log/fileLogLevel', type=str, help='isaac sim logging arguments', default='', required=False)
    parser.add_argument('--/log/outputStreamLevel', type=str, help='isaac sim logging arguments', default='', required=False)

    return parser

#Parser
parser = make_parser()
args = parser.parse_args()
head = args.headless
gripper_ID = args.gripper_ID
print(args.controller)

#launch Isaac Sim before any other imports
from omni.isaac.kit import SimulationApp
config= {
    "headless": head,
    'max_bounces':0,
    'fast_shutdown': True,
    'max_specular_transmission_bounces':0,
    'physics_gpu': args.device,
    'active_gpu': args.device
    }
simulation_app = SimulationApp(config) # we can also run as headless.


#World Imports
from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.cloner import GridCloner    # import Cloner interface
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import DynamicSphere, GroundPlane, VisualSphere

# Custom Classes
from managers import Manager
from views import View

#Omni Libraries
from omni.isaac.core.utils.stage import add_reference_to_stage,open_stage, save_stage
from omni.isaac.core.prims.rigid_prim import RigidPrim 
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import get_prim_children, get_prim_path, get_prim_at_path
from omni.isaac.core.utils.transformations import pose_from_tf_matrix
from omni.isaac.core.materials import PhysicsMaterial
import omni.isaac.core.utils.prims as prim_utils

def import_gripper(work_path,usd_path, EF_axis):
        """ Imports Gripper to World

        Args:
            work_path: prim_path of workstation
            usd_path: path to .usd file of gripper
            EF_axis: End effector axis needed for proper positioning of gripper
        
        """
        T_EF = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
        if (EF_axis == 1):
            T_EF = np.array([[ 0,0,1,0],
                            [ 0,1,0,0],
                            [-1,0,0,0],
                            [0,0,0,1]])
        elif (EF_axis == 2):
            T_EF = np.array([[1, 0,0,0],
                                [0, 0,1,0],
                                [0,-1,0,0],
                                [0, 0,0,1]])
        elif (EF_axis == 3):
            T_EF = np.array([[1, 0, 0,0],
                                [0,-1, 0,0],
                                [0, 0,-1,0],
                                [0, 0, 0,1]])
        elif (EF_axis == -1):
            T_EF = np.array([[0,0,-1,0],
                                [0,1, 0,0],
                                [1,0, 0,0],
                                [0,0, 0,1]])
        elif (EF_axis == -2):
            T_EF = np.array([[1,0, 0,0],
                                [0,0,-1,0],
                                [0,1, 0,0],
                                [0,0, 0,1]])
        #Robot Pose
        gripper_pose= pose_from_tf_matrix(T_EF.astype(float))
        # Adding Robot usd
        add_reference_to_stage(usd_path=usd_path, prim_path=work_path+"/gripper")
        
        robot = world.scene.add(Articulation(prim_path = work_path+"/gripper",
                            position = gripper_pose[0], orientation = gripper_pose[1], enable_dof_force_sensors = True))
        print(robot, usd_path)
        #robot.set_enabled_self_collisions(False)
        return robot, T_EF

def import_objects(work_path, num_w, step_size, init_r):
    """ Import Object .usd to World

    """
    material = PhysicsMaterial(prim_path = '/World/PhysicsMaterial_x',name = 'pm', static_friction =2.0, dynamic_friction =2.0)
    radii = []
    prims = []
    for i in range(num_w):
        r = init_r+(step_size*i)
        prim = DynamicSphere(prim_path= work_path[:-1]+str(i)+"/Sphere", color=np.array([1.0, 1.0, 0.0]), 
                            mass=0.1,radius =r, name = "Sphere"+str(i) , translation = [0,0,0], physics_material = material)
        #object_parent = world.scene.add(prim)
        prim.set_collision_approximation("convexHull")
        
        radii +=[r]
        prims+=[prim]
    #print(object_parent)

    return radii,prims


if __name__ == "__main__":
    
    # Directories
    grippers_directory = args.gripper_dir
    output_directory = args.output_dir
    
    if not os.path.exists(grippers_directory):
        raise ValueError("Grippers directory not given correctly")
    elif not os.path.exists(output_directory): 
        raise ValueError("Output directory not given correctly")

    # Testing Hyperparameters
    test_time = args.test_time
    controller = args.controller
    num_w = args.num_w
    step_size = args.step_size
    init_r = args.init_r

    #physics_dt = 1/120
    world = World(set_defaults = False)
    
    #Debugging
    render = not head
    out_name = gripper_ID + "_results.json"
    out_path = os.path.join(output_directory,out_name)

    # Initialize Manager
    manager = Manager(grippers_directory,gripper_ID, controller)   
    work_path = "/World/Workstation_0"
    work_prim = define_prim(work_path)

    robot, T_EF = import_gripper(work_path, manager.gripper_path,manager.gripper_dict["EF_axis"])
    #Clone
    cloner = GridCloner(spacing = 1)
    target_paths = []
    for i in range(num_w):
            target_paths.append(work_path[:-1]+str(i))
    cloner.clone(source_prim_path = "/World/Workstation_0", prim_paths = target_paths,
                     copy_from_source = True, base_env_path = "/World",
                     root_path = "/World/Workstation_")


    spheres_radii, prims = import_objects(work_path, num_w,step_size, init_r)

    plane = GroundPlane(prim_path="/World/GroundPlane", z_position=-1)

    contact_names = []
    for i in manager.gripper_dict["contact_names"]:
        contact_names.append(work_path[:-1]+"*"+"/gripper/" +  i)

    viewer = View(work_path, contact_names, num_w,manager, world, test_time, spheres_radii)


    light_1 = prim_utils.create_prim(
        "/World/Light_1",
        "DomeLight",
        attributes={
            "inputs:intensity": 1000
        }
    )

    
    #Reset World and create set first robot positions
    world.reset()

    # Set desired physics Context options
    #world.reset()
    physicsContext = world.get_physics_context()
    physicsContext.set_solver_type("TGS")
    physicsContext.set_physics_dt(1/manager.gripper_dict["physics_frequency"])
    physicsContext.enable_gpu_dynamics(True)
    physicsContext.enable_stablization(True)
    physicsContext.set_gravity(-10)
    viewer.grippers.initialize(world.physics_sim_view)
    viewer.objects.initialize(world.physics_sim_view)

    world.reset()
    print(robot.dof_names)
    #Post Reset
    viewer.grippers.initialize(world.physics_sim_view)
    viewer.objects.initialize(world.physics_sim_view)
    viewer.post_reset()

    

    #General Vars for saving
    manager.results["controller"]= viewer.controller.type
    manager.results["test_time"]= test_time
    manager.results["contact_names"]= manager.gripper_dict["contact_names"]

    #world.pause()    
    #Run Sim
    i=0
    with tqdm(total=len(viewer.completed)) as pbar: 
        while not all(viewer.completed):
            #print(mass)
            
            world.step(render=render) # execute one physics step and one rendering step if not headless
            if i == 0:
                world.pause()
                i+=1
            #world.pause()
            if pbar.n != np.sum(viewer.completed): #Progress bar
                pbar.update(np.sum(viewer.completed)-pbar.n)


    #Save new json with results
    manager.save_json(out_path)
    manager.print_results()

    simulation_app.close() # close Isaac Sim
        
