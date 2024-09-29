#External Libraries
import numpy as np
import time

#Custom Classes and utils
from utils import te_batch,re_batch

#Omni Libraries
from omni.isaac.core.utils.numpy.rotations import quats_to_rot_matrices
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.prims.rigid_prim import RigidPrimView, RigidPrim
from omni.isaac.core.articulations import  ArticulationView
from omni.isaac.core.utils.transformations import pose_from_tf_matrix, tf_matrices_from_poses
from omni.isaac.core.prims import XFormPrimView
from controllers import controller_dict


class View():
    """ISAAC SIM VIEWS Class 
    Facilitates the probing and the programming of the simulation. In this class you will find all the code of the simulation behavior.

    Args:
        work_path: prim_path of workstation prim
        contact_names_expr: Names of gripper meshes to filter collisions from in the Isaac Sim format
        num_w: Total number of workstations
        manager: Manager class containing grasp information
        world: Isaac Sim World object
        test_time: Total test time of each test
        mass: Mass of the object to test
    """
    def __init__(self, work_path, contact_names_expr, num_w,  manager, world, test_time, spheres_radii):

        #Create Views
        self.objects = world.scene.add(
            RigidPrimView(
                prim_paths_expr= work_path[:-1]+"*"+"/Sphere", 
                track_contact_forces = True, 
                prepare_contact_sensors = True, 
                contact_filter_prim_paths_expr  = contact_names_expr, 
                max_contact_count = len(contact_names_expr)))
        self.grippers = world.scene.add(
            ArticulationView(
                prim_paths_expr = work_path[:-1]+"*"+"/gripper",
                reset_xform_properties = False))
        
        # Initialize Variables
        self.num_w = num_w
        self.test_time = test_time
        self.world = world

        ws_poses = self.grippers.get_world_poses()
        self.ws_Ts = tf_matrices_from_poses(ws_poses[0],ws_poses[1])
        self.current_times = np.zeros((num_w,))
        self.grasp_set_up = np.zeros((num_w,))

        self.current_times = np.zeros((self.num_w,))
        self.set_up_timer = np.zeros((self.num_w,))
        self.radii = np.zeros((self.num_w,))
        self.contact_offsets = np.zeros((self.num_w,))
        self.grasp_set_up = np.zeros((self.num_w,))
        self.completed = np.zeros((self.num_w,))
        self.initial_check = np.zeros((self.num_w,))
        self.work_path= work_path


        self.radii = np.array(spheres_radii)
        #self.objects.enable_gravities()
        self.manager = manager
        self.gripper_ID = manager.gripper
        self.gripper_dict = manager.gripper_dict
        self.dofs = self.gripper_dict['opened_dofs']
        self.pose = self.gripper_dict['palm_pose']
        self.pose = np.array(self.pose)
        self.base_controller = controller_dict["transfer_default"]
        self.contact_th = self.gripper_dict["contact_th"]
        
        #Add physics Step
        world.add_physics_callback("physics_steps", callback_fn=self.physics_step)
        

    
    def post_reset(self):
        """ Code that needs to run after the reset of the world (dependent on the existence of physics context object)"""
        # Set grippers dofs
        self.grippers.set_joint_positions(self.dofs)

        # Calculate objects positions
        self.s_poses = np.zeros((self.num_w, len(self.pose)))
        for i in range(self.num_w):
            self.s_poses[i] = self.pose

        object_Ts = tf_matrices_from_poses(self.s_poses[:,:3], self.s_poses[:,3:])
        R = object_Ts[:,:3,:3]
        R_T = np.transpose(R,axes=[0,2,1])
        p = np.expand_dims(object_Ts[:,:3,3],axis=2)
        p = -1*np.matmul(R_T,p)
        object_Ts = np.concatenate((R_T,p), axis =2)
        tmp = np.zeros((object_Ts.shape[0],1, 4))
        tmp[:,:,3]=1
        object_Ts = np.concatenate((object_Ts,tmp), axis =1)

        object_Ts= np.matmul(self.ws_Ts, object_Ts)
        self.init_positions=np.zeros((object_Ts.shape[0],3))
        self.init_rotations =np.zeros((object_Ts.shape[0],4))
        for i in range(object_Ts.shape[0]):
            self.init_positions[i], self.init_rotations[i] = pose_from_tf_matrix(object_Ts[i].astype(float))
            self.init_positions[i] -= np.array([0,0,self.radii[i]])
        
        # Set object position and velocities
        self.objects.set_velocities([0,0,0,0,0,0]) 
        #self.objects_parents.set_world_poses(self.init_positions, self.init_rotations)
        self.objects.set_world_poses(self.init_positions, self.init_rotations)

        # Get max efforts and dofs
        dc = self.world.dc_interface
        articulation = dc.get_articulation(self.work_path+"/gripper")
        self.dof_props = dc.get_articulation_dof_properties(articulation)
        self.close_positions = np.zeros((self.num_w,len(self.dofs)))
        close_mask = self.gripper_dict["transfer_close_dir"]

        for i in range(len(self.dof_props)):
            if (close_mask[i]==0):
                self.close_positions[:,i]=(self.dofs[i])
            elif (close_mask[i]>0):
                self.close_positions[:,i]=(self.dof_props[i][3])
            elif (close_mask[i]<0):
                self.close_positions[:,i]=(self.dof_props[i][2]) 
            else: 
                raise ValueError("clos_dir arrays for grippers can only have 1,-1 and 0 values indicating closing direction")
        self.controller = self.base_controller(close_mask, self.close_positions)
        self.test_type = self.controller.type
        self.new_dofs = np.zeros((self.num_w,len(self.dofs)))
        self.final_gt = np.zeros((self.num_w,3))
        self.final_gq = np.zeros((self.num_w,4))
        self.final_st = np.zeros((self.num_w,3))
        self.final_sq = np.zeros((self.num_w,4))
        self.data = []
        self.last_rb_ind = []
        self.matrices = []
        for i in range(self.num_w):
            self.data+=[None]
            self.matrices+=[None]
        return
    
    def physics_step(self,step_size):
        """ Function runs before every physics frame

        step_size: time since last physics step. Depends on physics_dt
        """
        initial_ind = np.argwhere(self.initial_check==0) #ws indices
        if(len(initial_ind)>0):
            # Calculate falls
            tmp = np.count_nonzero(np.sum(np.squeeze(self.objects.get_contact_force_matrix(initial_ind)),axis =2),axis=1)

            finish_ind = initial_ind[tmp>=self.gripper_dict["init_contact_th"]]
            if(len(finish_ind)>0):
                self.test_failed(finish_ind)
            self.initial_check[initial_ind]=1

        active_ind = np.argwhere(self.completed==0) #ws indices
        if(len(active_ind)>0):
            # Calculate falls
            current_positions, current_rotations = self.objects.get_world_poses(active_ind)
            t_error = abs(te_batch(self.init_positions[active_ind], current_positions))

            finish_ind = active_ind[t_error>0.3]
            if(len(finish_ind)>0):
                self.test_failed(finish_ind)

        # Grasp Set Up
        tmp_active = np.squeeze(self.completed==0)
        rb_ind = np.argwhere(np.multiply(np.squeeze((self.grasp_set_up==0)),tmp_active) ==1)[:,0]
        if (len(rb_ind)>0):
            
            self.objects.set_velocities([0,0,0,0,0,0],rb_ind) 
            self.objects.set_world_poses(self.init_positions[rb_ind], self.init_rotations[rb_ind],rb_ind)
            tmp = np.count_nonzero(np.sum(self.objects.get_contact_force_matrix(rb_ind),axis =2),axis=1)
            
            #print(tmp)
            #Update grasp_setup
            self.current_times[rb_ind[tmp<self.contact_th]]=0
            self.grasp_set_up[rb_ind[tmp>=self.contact_th]]=1

            self.set_up_timer[rb_ind] +=step_size

            # Update sphere information
            self.new_dofs[rb_ind[tmp>=self.contact_th]] = self.grippers.get_joint_positions(indices= rb_ind[tmp>=self.contact_th])
            self.final_gt[rb_ind[tmp>=self.contact_th]], self.final_gq[rb_ind[tmp>=self.contact_th]] = self.grippers.get_world_poses(rb_ind[tmp>=self.contact_th])
            self.final_st[rb_ind[tmp>=self.contact_th]], self.final_sq[rb_ind[tmp>=self.contact_th]] = self.objects.get_world_poses(rb_ind[tmp>=self.contact_th])
            for ind in rb_ind:
                self.matrices[ind] = self.objects.get_contact_force_matrix(ind)
                self.data[ind] = self.objects.get_contact_force_data(ind)
        #Debug
        if rb_ind.tolist()!=self.last_rb_ind:
            print(rb_ind,"--", self.last_rb_ind)
            #print(self.objects.get_contact_force_matrix(rb_ind))
            #self.world.pause()
            pass
        self.last_rb_ind= rb_ind.tolist()
        #print(self.current_times)
        # Apply actions
        actions = self.controller.forward(self.manager.gripper, self.current_times, self.grippers, self.close_positions)
        self.grippers.apply_action(actions)

        # Update time
        self.current_times += step_size
        
        # End of testing time
        time_ind = np.argwhere(np.multiply(np.squeeze((self.current_times>self.test_time)),tmp_active))[:,0]
        if (len(time_ind)>0):
            tmp = np.count_nonzero(np.sum(self.objects.get_contact_force_matrix(time_ind),axis =2),axis=1)
            passed_ind = time_ind[tmp>=self.contact_th]
            failed_ind = time_ind[tmp<self.contact_th]
            #Update grasp_setup
            if (len(passed_ind)>0):
                self.test_finish(passed_ind)
            else:
                self.test_failed(failed_ind)

        #set_up fail
        set_up_ind = np.argwhere(np.multiply(np.squeeze((self.set_up_timer>self.test_time)),tmp_active))[:,0]
        if(len(set_up_ind)>0):
            self.test_failed(set_up_ind)
        return
    
    def test_finish(self, finish_ind):
        """ Function to reset workstations after tests are finished
        
        Args:
            finished_ind: IDs of Workstations that completed the test.
        """
        finish_ind=np.atleast_1d(np.squeeze(finish_ind))
        self.completed[finish_ind] = 1
        print("Passed: ", finish_ind)
        #Report Fall
        for ind in finish_ind:
            self.manager.report_passed(self.radii[ind],
                                    self.new_dofs[ind], self.data[ind], self.final_st[ind],
                                    self.final_sq[ind], self.final_gt[ind],
                                    self.final_gq[ind], self.matrices[ind])

        return

    def test_failed(self,finish_ind):
        finish_ind=np.atleast_1d(np.squeeze(finish_ind))
        print("Failed: ", finish_ind)
        self.completed[finish_ind] = 1
        for ind in finish_ind:
            self.manager.report_failed(self.radii[ind])
        return
