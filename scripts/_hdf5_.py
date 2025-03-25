import h5py

ROOT_DIR = "/home/ani/astar/my_Isaac/episodes"

# Read structure
def read_structure(path):
    """
    Read the data structure of h5 files.
    """
    with h5py.File(path, 'r') as f:
        def print_hdf5_structure(name, obj):
            print(name)
        f.visititems(print_hdf5_structure)


def read_values(path, key):
    """
    Read the key values. 
    """
    with h5py.File(path, 'r') as f:
        dset = f[key]
        print("Dataset shape:", dset.shape)
        print("First 10 values:", dset[4:5])

if __name__ == "__main__":

    for _ in range(4,5):
        path = ROOT_DIR + f"/episode_{_}.h5"
        # print("======")
        # read_values(path, "action")
        # read_structure(path)
        print("======")
        print("D_max")
        read_values(path, "D_max")
        print("======")
        print("D_min")
        read_values(path, "D_min")
        print("======")
        print("rbg values")
        read_values(path, "in_hand/rgb")
        print("======")
        print("depth values")
        read_values(path, "in_hand/depth")
        # print("======")
        # read_values(path, "index")
        # print(_)
        # print("======")
        # read_values(path, "label")