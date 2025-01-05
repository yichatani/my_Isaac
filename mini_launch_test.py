from omni.isaac.lab.app import AppLauncher

# launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app


########################## check if the module could be found
import sys
print("\n".join(sys.path))
##########################



from omni.kit.app import get_app
extension_manager = get_app().get_extension_manager()

extension_name = "omni.isaac.motion_generation"


if not extension_manager.is_extension_enabled(extension_name):
    extension_manager.set_extension_enabled(extension_name, True)
    print(f"Activated extension: {extension_name}")
else:
    print(f"Extension {extension_name} is already active!")


dependencies = ["omni.isaac.core", "omni.isaac.kit", "omni.ui"]

for dependency in dependencies:
    if not extension_manager.is_extension_enabled(dependency):
        extension_manager.set_extension_enabled(dependency, True)
        print(f"Activated dependency: {dependency}")


import omni.isaac.motion_generation

print("Import successfully! ")



# # keep running
# while simulation_app.is_running():
#     simulation_app.update()