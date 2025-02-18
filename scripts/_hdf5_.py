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
        print("First 10 values:", dset[50:100])

if __name__ == "__main__":

    path = ROOT_DIR + "/episode_0.h5"
    read_structure(path)
    print("======")
    read_values(path, "agent_pos")
    print("======")
    read_values(path, "action")
