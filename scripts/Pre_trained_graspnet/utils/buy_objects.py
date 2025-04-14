# summary objects id in test sets

import os

scenes_path = '/media/wws/Elements/Graspnet_1B/scenes'
store_id_list = list()

for i in range(160, 190):
    scene_path = os.path.join(scenes_path, 'scene_'+str(i).zfill(4))
    object_id_path = os.path.join(scene_path, 'object_id_list.txt')
    with open(object_id_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line not in store_id_list:
                store_id_list.append(line)

id_list = [int(x) for x in store_id_list]
id_list = sorted(id_list)
print('Finish')

