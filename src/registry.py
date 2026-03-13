import yaml
import importlib.util
import sys
import os

class ModelRegistry:
    def __init__(self, yaml_path="registry.yaml"):
        self.yaml_path = yaml_path
        self.models = self._load_yaml()

    def _load_yaml(self):
        """Parses the YAML file and maps models by their nickname."""
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"Registry file not found: {self.yaml_path}")
        
        with open(self.yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            # Convert the list of models into a dictionary keyed by 'name' for quick lookup
            return {m['name']: m for m in data.get("available_models", [])}

    def load_model(self, model_name):
        """Dynamically loads a python script as a module and returns it with its config."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not registered in {self.yaml_path}.")
        
        config = self.models[model_name]
        module_path = config["path"]
        
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Script missing: Cannot find {module_path}")

        # Create a safe, unique namespace for the dynamic module
        module_name_clean = f"dynamic_model_{model_name.replace('-', '_')}"
        
        # The standard Python importlib dance for loading from a file path
        spec = importlib.util.spec_from_file_location(module_name_clean, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name_clean] = module
        spec.loader.exec_module(module)
        
        return module, config