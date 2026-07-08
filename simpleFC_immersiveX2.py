import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import GC as g
import open3d as o3d
import math
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager as font_manager
import time
import csv
import os
import sys
import glob

from config import Config
from dataset import PcdDataset
from torch.utils.data import DataLoader

#global voxel creation
#redwood
upper_bound = 3.0
lower_bound = -1.8
gridStep = 0.02

#own dataset
# upper_bound = 3.9
# lower_bound = -2.8
# gridStep = 0.02

start_compress= time.time()
Clookup, Gvoxel, grid = g.funcCompressPC(lower_bound,upper_bound,gridStep)
end_compress= time.time()


def voxel_select(queries, global_voxel):
    v= np.asarray([global_voxel.get_voxel(pt) for pt in queries])
    return v

def get_color(occ_voxel,Clookup):
    colorlist =[]
    ylen = max(Clookup)[1]+1
    zlen = max(Clookup)[2]+1
    for pt in occ_voxel:
        colorlist.append([((pt[0]*ylen+pt[1])*zlen+pt[2])])

    #colorlist = [np.where((Clookup[:,0]==vt[0]) &(Clookup[:,1]==vt[1])&(Clookup[:,2]==vt[2]))[0][0] for vt in occ_voxel]
    return colorlist

def reconstruct(sent_voxel,global_voxel,i,pc):
    voxel_centr = np.array([global_voxel.get_voxel_center_coordinate(pt) for pt in sent_voxel])
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(voxel_centr)
    re_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(p, voxel_size=0.04)
    # o3d.visualization.draw_geometries([re_voxel])
    # alpha = 0.15
    # print(f"alpha={alpha:.3f}")
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(p, alpha)
    # mesh.compute_vertex_normals()
    #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    #o3d.io.write_voxel_grid('pair/v'+str(pc)+'_'+str(i+1)+'.ply',re_voxel)
    re_voxel = np.asarray([pt.grid_index for pt in re_voxel.get_voxels()])
    return  re_voxel

def getfile(config: Config, split:str):
    assert split in ["train", "val", "test"], "Invalid split, must be 'train', 'val' or 'test'"
    file_list = open(os.path.join(config.data_split_path, f"{split}.txt")).readlines()
    file_list = [os.path.join(config.data_path, f.strip()) for f in file_list]
    print(f"Found {len(file_list)} {split} files")
    return file_list

# total_time=[]
# avg_time = []
# avg_compression_rate = []
# avg_Bcompression_rate = []
# avg_comp_time = []
# avg_trans_time = []

compression_rate = []
Anncompression_rate = []
Snncompression_rate = []
enetime = []
comtime = []
decomtime = []
transtime = []
rmse =[]
avg_num=0



# DATA_GLOB   = "all/frame*.ply"  # e.g., frame000.ply ... frame114.ply
# files = sorted(glob.glob(DATA_GLOB))
# assert len(files) > 1, f"No files found with pattern {DATA_GLOB}"
i=0
config = Config()
test_dataset = getfile(config=config, split="val")

for pc in test_dataset:
    i+=1
    print("object ",i)
    #input

    pcd = o3d.io.read_point_cloud(pc)
    #o3d.visualization.draw_geometries([pcd])
    queries = np.divide(np.asarray(pcd.points, dtype=np.float32),100)
    queries =np.round(queries,2)
    pcd.points = o3d.utility.Vector3dVector(queries)



    voxel= o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.04)
    val = np.asarray([pt.grid_index for pt in voxel.get_voxels()])
    #print(len(pcd.points[0]))
    #print(val)
    #o3d.visualization.draw_geometries([voxel])

    # print("min value",i,np.min(queries))
    # print("max value", i, np.max(queries))

    #lookup time->
    start_lookup = time.time()
    occ_voxels = voxel_select(queries,Gvoxel)
    occ_voxels = np.unique(occ_voxels, axis=0)
    #print("occupied voxels",i, occ_voxels)
    colors = get_color(occ_voxels,Clookup)
    print("col",i, len(colors))
    #<-
    end_lookup = time.time()

    #packet

    raw_array = np.array(queries, dtype=np.float32)
    byte_view = raw_array.view(np.uint8)           # shape (n, 12) — 4 bytes per float
    rawbit = np.unpackbits(byte_view, axis=1)           # shape (n, 96) — 32 bits per float
    rawbit = rawbit.flatten()
    # rawbit = raw_array.size * 32  # 32 bits per float32
    # rawbit = np.unpackbits(np.array(queries, dtype=np.uint8), axis=1,count=64*3)
    # rawbit = rawbit.flatten()
    print(len(rawbit),"raw")

    # csbit = np.unpackbits(np.array(val, dtype=np.uint8), axis=1, count=64 * 3)
    # csbit = csbit.flatten()
    # cs_array = np.array(val, dtype=np.float32)
    # csbit = cs_array.size * 32  # 32 bits per float32
    # print(len(csbit),"cs")

    # bit = np.unpackbits(np.array(colors, dtype=np.uint8), axis=1,count=64)
    int_array = np.array(colors, dtype=np.float32)      # shape (86, 1)
    # np.set_printoptions(threshold=np.inf)
    byte_view = int_array.view(np.uint8)            # shape (86, 4) — 4 bytes per int32
    bit = np.unpackbits(byte_view, axis=1)          # shape (86, 32) — 32 bits per int32                  
    # print(bit, "SC")
    bit = bit.flatten()
    # print(len(bit),"SC")
    total_packet = len(bit)/(1460*8)


    #ubit = np.packbits(bit, axis=-1,bitorder='big').view(np.uint32)

    #entropy = g.entropy_count(len(Clookup)+1,colors)
    #print(len(colors), entropy)
    #min_col = set(colors)
    #print("pcd",i,len(min_col))

    comp_data_size = len(bit)
    comp_ratio = len(rawbit)/(comp_data_size)
    compression_rate.append(round(comp_ratio,2))
    print("cxr Compression rate: ",comp_ratio)

    Anncomp_ratio = len(rawbit)/((4096))
    Anncompression_rate.append(round(Anncomp_ratio,2))
    print("Ann Compression rate: ",Anncomp_ratio)

    Snncomp_ratio = len(rawbit)/((4096*9))
    Snncompression_rate.append(round(Snncomp_ratio,2))
    print("Snn Compression rate: ",Snncomp_ratio)

    #destination
    send_voxel=[]
    start_decod_lookup = time.time()
    for vc in colors:
        send_voxel.append(Clookup[vc[0]])
    send_voxel = np.array(send_voxel)
    re_voxel = reconstruct(send_voxel,Gvoxel,i,pc)
    end_decod_lookup = time.time()


    ttime = end_lookup-start_lookup+(total_packet*0.001)+0.02+end_decod_lookup-start_decod_lookup
    print("encoding/decoding time: ", end_lookup-start_lookup+end_decod_lookup-start_decod_lookup, "trans time: ", (total_packet*0.001)+0.02)
    print("end to end time for SC: ",i,"compression time (offline): ", end_compress-start_compress,"s online: ", ttime)
    avg_num = avg_num+len(re_voxel)
    comtime.append(end_lookup-start_lookup)
    transtime.append((total_packet*0.001)+0.001)
    decomtime.append(end_decod_lookup-start_decod_lookup)
    enetime.append(ttime)

    # avg_compression_rate.append(compression_rate)
    # avg_Bcompression_rate.append(Bcompression_rate)
    # #avg_time.append(enetime)
    # avg_comp_time.append(comtime)
    # avg_trans_time.append(transtime)
    #total_time.append(sum(enetime))
print("Sum for 627 pcd: ", sum(enetime))



with open('SC_time_4voxel.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(enetime)


with open('Cxrcomp_4voxel.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(compression_rate)

with open('Anncomp_4voxel.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(Anncompression_rate)

with open('Snncomp_4voxel.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(Snncompression_rate)


with open('cxr_comTime_4voxel.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(comtime)

with open('cxr_decomTime_4voxel.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(decomtime)

with open('cxr_transTime_4voxel.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(transtime)
