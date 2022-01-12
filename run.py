import main
import subprocess
subprocess.call("py -3.7 -m pip install tensorflow")
subprocess.call("py -3.7 -m pip install opencv-python")
subprocess.call("py -3.7 -m pip install numpy")
subprocess.call("py -3.7 -m pip install keras")

if __name__ == '__main__':
    main.main()