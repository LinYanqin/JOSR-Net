import pickle
import tensorlayer as tl
import numpy as np
import shutil
import os
import random

data_path = "./JOSRdata/"
training_gooddata_path = str(data_path)+"Label"
val_ratio = 0.2

# ===================================   =================================== #
# save data into pickle format

f_traingood_all = tl.files.load_file_list(path=training_gooddata_path,
                                          regx='.*.txt',
                                          printable=False)
f_traingood_all.sort(key=tl.files.natural_keys)
traingood_all_num = len(f_traingood_all)
valgood_num = int(traingood_all_num * val_ratio)

f_traingood = []
f_valgood = []

random.seed(42)
val_idex = random.sample(range(0, traingood_all_num-1), valgood_num)
for i in range(traingood_all_num):
    if i in val_idex:
        f_valgood.append(os.path.join(training_gooddata_path, f_traingood_all[i]))
    else:
        f_traingood.append(f_traingood_all[i])

traingood_num, valgood_num = len(f_traingood), len(f_valgood)

print(f"{valgood_num} need to transform ")

# ===================================   =================================== #
for i in range(valgood_num):
    shutil.move(f_valgood[i], os.path.join(str(data_path), "ValidationGoodData/"))

# ===================================   =================================== #
print("processing data finished!")
