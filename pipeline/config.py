from dataclasses import dataclass, field
from typing import List

@dataclass
class PipelineConfig:
    # IO
    input_dir: str = "data/raw"
    work_dir: str = "outputs/run1"
    channel_mode: str = "split"  # "split", "multi", or "single"
    r_suffix: str = "_R"
    g_suffix: str = "_G"
    image_exts: List[str] = field(default_factory=lambda: [".tif", ".tiff", ".png", ".jpg"])

    # Segmentation
    segmenter: str = "otsu"
    min_area: int = 150
    max_area: int = 999999
    gaussian_sigma: float = 1.0

    # Crops
    crop_size: int = 150

    # Features
    distances: List[int] = field(default_factory=lambda: [1, 2, 4])
    angles: List[float] = field(default_factory=lambda: [0, 0.785398, 1.570796, 2.356194])
    lbp_radius: int = 2
    lbp_n_points: int = 16
    hog_pixels_per_cell: int = 16
    hog_cells_per_block: int = 2
    gabor_frequencies: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    gabor_thetas: List[float] = field(default_factory=lambda: [0, 0.785398, 1.570796, 2.356194])

    # Modeling
    rf_n_estimators: int = 400
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 2
    cv_folds: int = 10
    random_state: int = 42
    n_jobs: int = -1

    # CNN (optional)
    enable_cnn: bool = False
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    focal_loss_gamma: float = 2.0
