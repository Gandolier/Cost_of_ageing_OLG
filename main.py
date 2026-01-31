import importlib
import sys
import types
import os
import inspect


def reimport():
    """
    Finds all local modules imported in the caller's namespace (e.g. notebook),
    reloads them, and updates the specific functions/variables imported from them.
    """

    # 1. Get the caller's globals (the notebook's namespace), not this module's globals
    caller_frame = inspect.currentframe().f_back
    current_globals = caller_frame.f_globals

    # Determine the project root (directory containing this utils.py file)
    # We will only reload modules that live inside this directory.
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 2. Identify which modules need reloading
    modules_to_reload = set()

    for name, val in current_globals.items():
        if name.startswith('_'): continue

        # Case A: It's a module object
        if isinstance(val, types.ModuleType):
            modules_to_reload.add(val)

        # Case B: It's an object from a module (function, class, etc.)
        elif hasattr(val, '__module__') and val.__module__ in sys.modules:
            module = sys.modules[val.__module__]
            modules_to_reload.add(module)

    # 3. Reload modules, filtering for LOCAL modules only
    reloaded_modules = {}

    for module in modules_to_reload:
        mod_name = module.__name__

        # Skip modules without a file path (built-ins)
        if not hasattr(module, '__file__') or module.__file__ is None:
            continue

        mod_file = module.__file__

        # CRITICAL FIX: strict filtering
        # 1. Must be inside the project root directory
        if not mod_file.startswith(project_root):
            continue

        # 2. Must not be in site-packages (in case venv is inside project root)
        if 'site-packages' in mod_file or 'dist-packages' in mod_file:
            continue

        # Reload
        try:
            new_module = importlib.reload(module)
            reloaded_modules[mod_name] = new_module
        except Exception as e:
            print(f"   [ERROR] Could not reload '{mod_name}': {e}")

    # 4. Update references in the notebook namespace
    count = 0
    for name, val in current_globals.items():
        if name.startswith('_'): continue

        if hasattr(val, '__module__') and val.__module__ in reloaded_modules:
            new_module = reloaded_modules[val.__module__]

            if hasattr(val, '__name__') and hasattr(new_module, val.__name__):
                new_val = getattr(new_module, val.__name__)

                if new_val is not val:
                    current_globals[name] = new_val
                    count += 1