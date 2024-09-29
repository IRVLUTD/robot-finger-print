# GenDexGrasp Tests using IsaacGym 
A forked version of the GenDexGrasp paper [repository](https://github.com/tengyu-liu/GenDexGrasp?tab=readme-ov-file) used for evaluating grasp generation methods. 

## Installation

- Download and install the Isaac Gym Preview 4 release from [Isaac Gym](https://developer.nvidia.com/isaac-gym), follow the installation steps to create a new conda environment (will simplify things).

- Install the requirements listed in requirements.yaml within the Isaac Gym conda environment using the following command:

```Shell
    conda install --file requirements.yaml
```

- Symlink the GenDexGrasp `data` folder to `./data/` so that `./data/` folder links to the `object` and `urdf` data directories from GenDexGrasp.

## Running tests
The `run_grasp_test.py` script is used to run the grasping tests described in the [GenDexGrasp paper](https://arxiv.org/abs/2210.00722), it is a slightly altered version of their script. Feel free to use their version since its practically the same except, the change in the simulation refinement step where we change the parameters for 1-step gradient descent with `step_size=0.02` and `learning_rate=0.1` which differs from the values used by GenDexGrasp.

The other change is that teh `run_grasp_test.py` consumes the grasp optimization results in a single folder which should simply things when testing grasps.

It takes the following arguments:

- robot_name: name of the gripper to be used 
- data_dir: folder path containing all the `grasps.pt` files to test (result from the grasp optimization)
- object_list: .json file path with list of objects to evaluate (just uses the objects under the 'validate' dictionary key).
- output_dir: directory save the results to.
- output_name: name of .json file where results will be printed.
- filtered: Boolean argument to signal if .pt contain all the generated grasps for an object gripper pair or if they have already been filtered (using minimum energy).
- headless: Boolean argument to run the simulation headless.

An example run command is shown below:

```sh
python run_grasp_test.py --filtered --headless \
--data_dir=[GRASP_OPT_RESULT_DIR] \
--robot_name=ezgripper \
--output_name=unseen_ezgripper
```


