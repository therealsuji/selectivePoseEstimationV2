import subprocess
import pathlib
import os
path = os.path.split(os.path.realpath(__file__))[0]
main_path = pathlib.Path(path).__str__()
main_path = main_path+"/outputfile.npy"
def generate_blend_file():
    myargs = [
        "C:/Program Files/Blender Foundation/Blender 2.90/blender.exe",
        "-b",
        "-P",
        "blender_importv2.py",
        "--",main_path
        ]
    subprocess.run(myargs)
