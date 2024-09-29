import argparse
import json
import os.path
import time
import sys
import shutil
from isaacgym import gymapi

import torch
from utils.set_seed import set_global_seed
import webbrowser
import yaml
import platform
import gc
from deepdiff import DeepHash
import trimesh as tm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='barrett', type=str)

    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--object_list', default='./split_train_validate_objects.json', type=str)
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--output_name', default='test_results', type=str)
    
    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--filtered', help='prefiltered results by energy',
                         default=False, action = 'store_true')
    parser.add_argument('--headless', help='Run simulation headless',
                         default=False, action = 'store_true')
    args_ = parser.parse_args()
    tag = str(time.time())
    return args_, tag


def get_sim_param():
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = 0
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.physx.num_subscenes = 0
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.num_threads = 0
    return sim_params


def compute_penetration(opt_q, object_name):
    from utils.get_models import get_handmodel
    import trimesh as tm
    import trimesh.sample
    # get object point cloud and normal cloud
    object_mesh: tm.Trimesh
    npts_object = 2048
    object_mesh = tm.load(os.path.join('data/ContactDB', object_name.split('+')[1],
                                       f'{object_name.split("+")[1]}_scaled.stl'))
    object_point_cloud, faces_indices = trimesh.sample.sample_surface(mesh=object_mesh, count=npts_object)
    object_normal_cloud = torch.tensor([object_mesh.face_normals[x] for x in faces_indices]).float().cuda()
    object_point_cloud = torch.Tensor(object_point_cloud).float().cuda()
    # object_point_cloud = torch.cat([object_point_cloud, contact_points_normal], dim=1).to(device)

    # get hand model
    num_particles = opt_q.shape[0]
    hand_model = get_handmodel(robot_name, opt_q.shape[0], device, hand_scale=1.)
    hand_model.update_kinematics(q=opt_q.clone().to(device))
    hand_surface_points_ = hand_model.get_surface_points()
    npts_hand = hand_surface_points_.size()[1]
    batch_object_point_cloud = object_point_cloud.unsqueeze(0).repeat(num_particles, 1, 1)
    batch_object_point_cloud = batch_object_point_cloud.reshape(num_particles, 1, npts_object, 3)
    hand_surface_points = hand_surface_points_.reshape(num_particles, 1, npts_hand, 3)
    hand_surface_points = hand_surface_points.repeat(1, npts_object, 1, 1).transpose(1, 2)
    batch_object_point_cloud = batch_object_point_cloud.repeat(1, npts_hand, 1, 1)
    hand_object_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
    hand_object_dist, hand_object_indices = hand_object_dist.min(dim=2)
    hand_object_points = torch.stack([object_point_cloud[x, :] for x in hand_object_indices], dim=0)
    hand_object_normal = torch.stack([object_normal_cloud[x, :] for x in hand_object_indices], dim=0)
    hand_object_signs = ((hand_object_points - hand_surface_points_) * hand_object_normal).sum(dim=2)
    hand_object_signs = (hand_object_signs > 0).float()
    penetration = (hand_object_signs * hand_object_dist).mean(dim=1)
    return penetration


if __name__ == '__main__':

    '''
    Sample Run:

    python run_grasp_test.py --filtered --headless --data_dir=gopt_results/fullrobots-sharp_lift_penw60/ood/ezgripper --robot_name=ezgripper --output_name=ez1
    '''


    set_global_seed(seed=42)
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=8)
    args, time_tag = get_parser()
    print(args)
    print(f'double check....')
    # time.sleep(2.)yiran

    cfg_path = './envs/tasks/grasp_test_force.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    adam_cfg_p ='envs/tasks/adam_config.yaml'
    with open(adam_cfg_p) as f:
        adam_cfg = yaml.safe_load(f)

    sim_params = get_sim_param()
    if args.robot_name == 'robotiq_3finger':
        sim_params.physx.contact_offset = 0.1

    time_tag = DeepHash(cfg)[cfg]
    # if args.robot_name == 'allegro':
    #     cfg_path = 'envs/tasks/grasp_test_force_allegro.yaml'
    #     with open(cfg_path) as f:
    #         cfg = yaml.safe_load(f)
    robot_name = args.robot_name
    if robot_name == 'shadowhand':
        from envs.tasks.grasp_test_force_shadowhand import IsaacGraspTestForce_shadowhand as IsaacGraspTestForce
    elif robot_name == 'barrett':
        from envs.tasks.grasp_test_force_barrett import IsaacGraspTestForce_barrett as IsaacGraspTestForce
    elif robot_name == 'ezgripper':
        from envs.tasks.grasp_test_force_ezgripper import IsaacGraspTestForce_ezgripper as IsaacGraspTestForce
    else:
        raise NotImplementedError

    sim_headless = args.headless
    device = "cuda"

    # load object list
    object_list = json.load(open(args.object_list))['validate']
    object_list.sort()

    
    data_basedir = args.data_dir
    record_path = os.path.join(args.output_dir, f'{args.output_name}.json') 
    tra_dir = data_basedir # Folder containing grasp data
    tra_path_list = os.listdir(tra_dir)

    isaac_model = None
    if args.mode == 'debug':
        pass
    elif args.mode == 'test':
        #load or create new record
        try:          
            test_record = json.load(open(record_path, 'rb'))
            old_object_list = object_list.copy()
            for object_name in old_object_list:
                if bool(test_record[object_name]):
                    object_list.remove(object_name)
            del old_object_list
            print(f'load record from: {record_path} ...')
            print(f'object list: {object_list}')
        except FileNotFoundError:
            print('create a new record')
            test_record = {x: {} for x in object_list}
            test_record['cfg'] = cfg
            test_record['adam_cfg'] = adam_cfg

        # Data to test
        data_listdir = os.listdir(tra_dir)
        if not args.filtered:
            data_listdir.sort(key=lambda x: int(x.split('-')[2].split('.pt')[0]))

        print(data_listdir)        
        
        for object_name in object_list:
            print(f'Test for {object_name}')
            q_tra_best = []
            #Get min energy grasp for testing
            
            if not args.filtered: 
                #GenDexGrasp loading and filtering of data
                for tra_path in data_listdir:
                    if tra_path.split('-')[1] != object_name:
                        continue
                    i_record = torch.load(os.path.join(tra_dir, tra_path))
                    q_tra = i_record['q_tra']
                    energy = i_record['energy']
                    q_tra_best.append(q_tra[energy.min(dim=0)[1], -1, :].unsqueeze(0).to(device))
                if len(q_tra_best)==0:
                    print(f"Object {object_name} was not found within the grasps provided.")
                    continue
                q_final_best = torch.cat(q_tra_best, dim=0)
            else:
                # If already filtered the minimum energy grasp
                for file in data_listdir:
                    if file.split('-')[2][:-3] != object_name:
                        continue
                    i_record = torch.load(os.path.join(tra_dir,file))
                    break
                q_final_best = i_record['q_data']
                    
            
            print(cfg['eval_policy'])
            if isaac_model is not None:
                del isaac_model
                gc.collect()

            # Load object
            object_mesh_path = f'./data/object/{object_name.split("+")[0]}/{object_name.split("+")[1]}/{object_name.split("+")[1]}.stl'
            object_mesh = tm.load(object_mesh_path)
            object_volume = object_mesh.volume
            print(f'object volume: {object_volume}')


            #Run Test
            isaac_model = IsaacGraspTestForce(cfg, sim_params, gymapi.SIM_PHYSX, "cuda", 0, headless=sim_headless,
                                              init_opt_q=q_final_best, object_name=object_name, object_volume=object_volume,
                                              fix_object=False)
            achieve_6dir = isaac_model.push_object()

            # Save result
            test_record[object_name][f'total_num'] = int(achieve_6dir.shape[0])
            test_record[object_name][f'succ_num'] = int(achieve_6dir.sum())
            test_record[object_name]['succ_flag'] = achieve_6dir.tolist()
            print(test_record[object_name])
            print(f'Is Grasp Stable: {achieve_6dir}')
            json.dump(test_record, open(record_path, 'w'))
    else:
        raise NotImplementedError
