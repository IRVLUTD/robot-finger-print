import json
from utils_model.HandModel import GenDexHandModel


def get_handmodel(robot, batch_size, device, model_type="gcs", hand_scale=1.):
    urdf_assets_meta = json.load(open("data/urdf/urdf_assets_meta.json"))
    urdf_path = urdf_assets_meta['urdf_path'][robot]
    meshes_path = urdf_assets_meta['meshes_path'][robot]
    if model_type == "gdx":
        hand_model = GenDexHandModel(robot, urdf_path, meshes_path, batch_size=batch_size, device=device, hand_scale=hand_scale)
    else:
        raise NotImplementedError
    return hand_model

