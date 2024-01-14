from .scene_gen import SceneGenDataset
from .your_own_data import YourOwnDataset



dataset_dict = {'scene_gen':SceneGenDataset,
                'own_data':YourOwnDataset}