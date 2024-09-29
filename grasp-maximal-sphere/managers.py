import numpy as np
import os
from controllers import controller_dict
import utils
import json
import time

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
class Manager:
    """ Grasp Data Manager:
    Manages the information of all grippers, grasps and the reporting of results (Verbosity and saving)
    This class takes in a specific .json structure, if it is desired to change the .json format, this 
    is the place to make the simulation compatible to the new format.

    Args:
        grasps_path: .json file containing the grasp information.
        grippers_path: folder path for all the gripper .usd folders
        objects_path: folder path for all the object .urdf files
        controller: specified controller to use in grippers
        world: Isaac Sim World object
    """
    def __init__(self, grippers_path, gripper_ID, num_w ,controller= 'default',
                 world=None):
        # Loading files and urdfs
        self.world = world
        grippers_info= os.path.join(grippers_path, "gripper_isaac_info.json")
        with open(grippers_info) as fd:
            grippers_dict = json.load(fd)
        
        self.gripper = gripper_ID #gripper IDs
        self.gripper_dict= grippers_dict[gripper_ID]
        
        # Check for usds Object's and Gripper's
        self._check_gripper_usd(grippers_path,gripper_ID)

        # Extract info from dictionaries external and internal
        self.controller = controller_dict[controller]
        self.init_time = time.time()
        
        #Pointer and result vars
        self.total_test_time = None #
        self.completed = np.zeros((len(num_w,)))
        #self.final_dofs = np.zeros((len(self.fall_time),len(self.close_mask)))
        
        self.results= dict()
        self.results["gripper"] = gripper_ID
        self.passed =[]
        self.failed = []


    def _check_gripper_usd(self,gripper_path, i):
        """ Check if the gripper usd exist

            grippers_path: Path to directory containing all gripper files. Make sure every gripper.urdf is within a folder with the same name
            i: gripper name
        """
        self.gripper_path = os.path.join(gripper_path, i, i, i+".usd")
        if (os.path.exists(self.gripper_path)) : 
            print("Found Gripper:", self.gripper_path )
        else: 
            raise LookupError("Couldn't find gripper .usd file at " +  self.gripper_path )
        return
    def report_passed(self, value, new_dofs, contact_data, sphere_t, sphere_q,
                      gripper_t, gripper_q, matrix):
        """ Reports results
        """
        # Decompress raw data
        normal_forces = contact_data[0]
        contact_points = contact_data[1]
        normals = contact_data[2]
        distances = contact_data[3]
        pair_contact_counts = contact_data[4]
        indices_pair_contact_cnt = contact_data[5]
        
        key = str(value)
        self.results[key] = dict()
        self.results[key]["status"] = "PASSED"
        self.results[key]["sphere_radius"] = value
        self.results[key]["dofs"] = new_dofs.tolist()
        self.results[key]["normal_forces"] = normal_forces.tolist()
        self.results[key]["contact_points"] = contact_points.tolist()
        self.results[key]["normals"] = normals.tolist()
        self.results[key]["distances"] = distances.tolist()
        self.results[key]["pair_contact_counts"] = pair_contact_counts.tolist()
        self.results[key]["indices_pair_contact_cnt"] = indices_pair_contact_cnt.tolist()
        self.results[key]["sphere_t"] = sphere_t.tolist()
        self.results[key]["sphere_q"] = sphere_q.tolist()
        self.results[key]["gripper_t"] = gripper_t.tolist()
        self.results[key]["gripper_q"] = gripper_q.tolist()
        #self.results[key]["contact_force_matrix"] = matrix.tolist()         
        #print("Gripper Reported", grippper_ID)
        self.passed += [value]
        return
    
    def report_failed(self,value):
        key = str(value)
        self.results[key] = dict()
        self.results[key]["status"] = "FAILED"
        self.results[key]["sphere_radius"] = value
        self.failed+=[value]
        return

    def save_json(self,output_path):
        """ Saves json on disk.

        Args: 
            out_path: path to save json at
        
        """
        print("Saving File at: ",output_path)

        with open(output_path,'w') as outfile:
            json.dump(self.results,outfile, cls=NpEncoder)
        return

    def print_results(self):
        """ Verbosity for results of .json file filter
        """
        print("Completed")
        print("Passed:", self.passed)
        print("Failed:", self.failed)
        
        return


