"""Offline replay entry point without importing the research package initializer."""
import importlib.util
from pathlib import Path
import sys


def load_tool(name):
    root=Path(__file__).resolve().parents[1]/'research/yank_deployed_validation'
    package='_yank_deployed_validation_cli'
    if package not in sys.modules:
        spec=importlib.util.spec_from_file_location(package,root/'__init__.py',submodule_search_locations=[str(root)])
        module=importlib.util.module_from_spec(spec);sys.modules[package]=module;spec.loader.exec_module(module)
    spec=importlib.util.spec_from_file_location(package+'.'+name,root/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def main():return load_tool('replay').main()

if __name__=='__main__':main()
