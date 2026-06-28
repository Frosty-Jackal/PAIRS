from torch.utils.data import Dataset
from face_replace.data.datasets.coach_dataset import CoachDataset


class PairedDataset(CoachDataset):
    """Paired image dataset used in debug/augmentation modes."""
    pass
