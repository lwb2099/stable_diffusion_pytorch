import argparse
from dataclasses import dataclass, field, fields
from omegaconf import DictConfig, OmegaConf


# Define data classes
@dataclass
class Train:
    a1: int = field(default=1)


@dataclass
class Val:
    a2: int = field(default=2)


# Create an argument parser
parser = argparse.ArgumentParser()

# Get a list of data classes
data_classes = [Train, Val]

# Create argument groups and arguments for each data class
for data_class in data_classes:
    group = parser.add_argument_group(data_class.__name__.lower())
    for field_info in fields(data_class):
        arg_name = "--" + field_info.name
        arg_type = field_info.type
        default_value = field_info.default
        group.add_argument(arg_name, type=arg_type, default=default_value)

# Parse the command-line arguments
args = parser.parse_args()

# Convert data classes to nested DictConfigs
cfg = DictConfig({})

for data_class in data_classes:
    data_instance = data_class()
    for field_info in fields(data_class):
        field_name = field_info.name
        field_value = getattr(args, field_name)
        setattr(data_instance, field_name, field_value)
    group_name = data_class.__name__.lower()
    group_config = OmegaConf.structured(data_instance)
    cfg[group_name] = group_config

# Access the arguments using the desired structure
print(cfg.train.a1)  # Output: 1
print(cfg.val.a2)  # Output: 2
