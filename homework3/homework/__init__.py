# Make homework a package
# Optional: expose commonly used modules at package level
from .models import load_model
from .datasets import SuperTuxClassificationDataset, load_data, get_class_names
