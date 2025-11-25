import math
import networkx as nx
import numpy as np
from statistics import mean
import open3d as o3d


def visualize(edge,color):
    G = nx.Graph()
    G.add_edges_from(edge)
    color_list = [color[node] for node in G.nodes()]
    nx.draw_networkx(G,node_color=color_list, with_labels=True)

def Edge(edge,a, b):
    temp = [a, b]
    edge.append(temp)

def uniaddEdge(adj, v, w): #i
    adj[v].append(w)

    # Note: the graph is undirected
    adj[w].append(v)
    return adj

def diraddEdge(adj, v, w): #i
    adj[v].append(w)

    return adj


# Assigns colors (starting from 0) to all
# vertices and prints the assignment of colors
def greedyColoring(adj, V):
    result = [-1] * V #ii

    # Assign the first color to first vertex
    result[0] = 0 #iii

    # A temporary array to store the available colors.
    # True value of available[cr] would mean that the
    # color cr is assigned to one of its adjacent vertices
    available = [False] * V #vii

    # Assign colors to remaining V-1 vertices
    for u in range(1, V):

        # Process all adjacent vertices and
        # flag their colors as unavailable
        for i in adj[u]:#iv
            if (result[i] != -1):
                available[result[i]] = True

        # Find the first available color
        cr = 0
        while cr < V: #v
            if (available[cr] == False):
                break
            cr += 1

        # Assign the found color
        result[u] = cr #vi

        # Reset the values back to false
        # for the next iteration
        for i in adj[u]: #vii
            if (result[i] != -1):
                available[result[i]] = False

    # Print the result
    '''for u in range(V):
        print("Vertex", u, " ---> Color", result[u])'''

    return max(result)+1, result


def hierarchyColoring(adj, V):
    result = [-1] * V #ii
    temp_adj = adj.copy()

    # Assign the first color to the vertex with the highest edges
    h_deg_idx = adj.index(max(adj,key=len))
    result[h_deg_idx] = 0 #iii

    h_array = []
    for i in range(len(temp_adj)):
        tmp = temp_adj.index(max(temp_adj,key=len))
        h_array.append(tmp)
        temp_adj[tmp]=[]

    available = [False] * V #vii

    # Assign colors to remaining V-1 vertices
    for u in h_array:
        if u==h_deg_idx:
            continue
        # Process all adjacent vertices and
        # flag their colors as unavailable
        for i in adj[u]:#iv
            if (result[i] != -1):
                available[result[i]] = True

        # Find the first available color
        cr = 0
        while cr < V: #v
            if (available[cr] == False):
                break
            cr += 1

        # Assign the found color
        result[u] = cr #vi

        # Reset the values back to false
        # for the next iteration
        for i in adj[u]: #vii
            if (result[i] != -1):
                available[result[i]] = False

    # Print the result
    '''for u in h_array:
        print("Vertex", u, " ---> Color", result[u])'''

    return max(result)+1, result


def entropy_count(cr, result): # number of color, assigned color list
    ent =0

    for i in range(cr):
        if result.count(i)!=0:
            temp_ent = (result.count(i)/len(result)) * math.log2(len(result)/result.count(i))
            ent = ent + temp_ent

    return ent



def funcCompress(x,y,func,dir="bi",algo="greedy",number_of_var =2): #max two inputs
    colorlookup = []
    combination = []
    g1 = []
    edges = []
    Cr_num = []
    colors = []
    entropy =[]
    length = [len(x),len(y)]
    #print(length)

    for i in range(number_of_var):
        temp_combination = [[] for gc in range(length[i])]
        temp_g1 = [[] for gc in range(length[i])]
        temp_edge = []
        for c in range(length[i%2]):  # table for possible valve values. %2 is to interchange between two lenghts. at first (for each x, all y). then (for each y, all x)
            for e in range(length[(i+1)%2]):
                if i:
                    v = func(x[e],y[c]) #v = math.ceil(ubias + Kc*errindex[ec] + ierrindex[iec]*Ki)
                else:
                    v = func(x[c], y[e])  # v = math.ceil(ubias + Kc*errindex[ec] + ierrindex[iec]*Ki)
                temp_combination[c].append(v)
        #print(temp_combination)

        for gc in range(length[i]):
            for jc in range(length[i]):
                if temp_combination[gc] == temp_combination[jc]:
                    continue
                else:
                    if dir == "uni":
                        temp_g1 = uniaddEdge(temp_g1, gc, jc)
                    elif dir == "bi":
                        temp_g1 = diraddEdge(temp_g1, gc, jc)
                    Edge(temp_edge, gc, jc)
        # print(g1)
        if algo == "hierarchy":
            temp_Cr_num, temp_colors = hierarchyColoring(temp_g1, length[i])  # graph coloring
        elif algo == "greedy":
            temp_Cr_num, temp_colors = greedyColoring(temp_g1, length[i])  # graph coloring
        temp_entropy = entropy_count(temp_Cr_num, temp_colors) # number of color, assigned color list
        print("chromatic number for var ",i,": ", temp_Cr_num, temp_colors, temp_entropy)
        combination.append(temp_combination)
        entropy.append(temp_entropy)
        g1.append(temp_g1)
        edges.append(temp_edge)
        Cr_num.append(temp_Cr_num)
        colors.append(temp_colors)

    #return Cr_num, colors, combination, g1Edge


    if number_of_var == 2:
        for xcr in range(Cr_num[0]):
            for ycr in range(Cr_num[1]):
                colorlookup.append([xcr, ycr, combination[0][colors[0].index(xcr)][colors[1].index(ycr)]])
    elif number_of_var == 1:
        for cr in range(Cr_num[0]):
            colorlookup.append([cr, combination[0][colors[0].index(cr)]])

    return colorlookup, colors, edges, entropy


def multilvlfuncCompress(x,y,func,dir="bi",algo="greedy",number_of_var =2): #max two inputs
    print(len(x)*len(x), len(y)*len(y))
    colorlookup = []
    combination = []
    g1 = []
    edges = []
    Cr_num = []
    colors = []
    length = [len(x),len(y)]

    for i in range(number_of_var):
        temp_combination = [[] for gc in range(length[i])]
        mul_combination = [[] for gc in range(length[i] * length[i])]
        lookup_combination = [[] for gc in range(length[i] * length[i])]
        temp_g1 = [[] for gc in range(length[i] * length[i])]
        temp_edge = []
        for ec in range(length[i%2]):  # table for possible valve values
            for iec in range(length[(i+1)%2]):
                if i:
                    v = func(x[iec],y[ec]) #v = math.ceil(ubias + Kc*errindex[ec] + ierrindex[iec]*Ki)
                else:
                    v = func(x[ec], y[iec])  # v = math.ceil(ubias + Kc*errindex[ec] + ierrindex[iec]*Ki)
                # print(v)
                temp_combination[ec].append(v)
        # print(valvelookup_err)

        for gc in range(length[i]):
            for jc in range(length[i]):
                mul_combination[jc + gc * length[i]].append(temp_combination[gc])
                mul_combination[jc + gc * length[i]].append(temp_combination[jc])



        for gc in range(length[i]*length[i]):
            for jc in range(length[i]*length[i]):
                if mul_combination[gc] == mul_combination[jc]:
                    continue

                elif mul_combination[gc][0] == mul_combination[jc][0]:
                    if mul_combination[gc][1] != mul_combination[jc][1]:
                        if dir == "uni":
                            temp_g1 = uniaddEdge(temp_g1, gc, jc)
                        elif dir == "bi":
                            temp_g1 = diraddEdge(temp_g1, gc, jc)

                elif mul_combination[gc][1] == mul_combination[jc][1]:
                    if mul_combination[gc][0] != mul_combination[jc][0]:
                        if dir == "uni":
                            temp_g1 = uniaddEdge(temp_g1, gc, jc)
                        elif dir == "bi":
                            temp_g1 = diraddEdge(temp_g1, gc, jc)
                elif mul_combination[gc][0] != mul_combination[jc][0]:
                    if mul_combination[gc][1] != mul_combination[jc][1]:
                        if dir == "uni":
                            temp_g1 = uniaddEdge(temp_g1, gc, jc)
                        elif dir == "bi":
                            temp_g1 = diraddEdge(temp_g1, gc, jc)
                else:
                    continue
                Edge(temp_edge, gc, jc)
        # print(g1)
        if algo == "hierarchy":
            temp_Cr_num, temp_colors = hierarchyColoring(temp_g1, length[i]*length[i])  # graph coloring
        elif algo == "greedy":
            temp_Cr_num, temp_colors = greedyColoring(temp_g1, length[i]*length[i])  # graph coloring
        entropy = entropy_count(temp_Cr_num, temp_colors)
        print("chromatic number for var ",i,": ", temp_Cr_num, temp_colors, entropy)

        for gc in range(length[i]):
            for jc in range(length[i]):
                for a in temp_combination[gc]:
                    for b in temp_combination[jc]:
                        #print(a,b)
                        lookup_combination[jc + gc * length[i]].append(max(a,b))

        combination.append(lookup_combination)
        g1.append(temp_g1)
        edges.append(temp_edge)
        Cr_num.append(temp_Cr_num)
        colors.append(temp_colors)


    if number_of_var == 2:
        for xcr in range(Cr_num[0]):
            for ycr in range(Cr_num[1]):
                colorlookup.append([xcr, ycr, combination[0][colors[0].index(xcr)][colors[1].index(ycr)]])
    elif number_of_var == 1:
        for cr in range(Cr_num[0]):
            colorlookup.append([cr, combination[0][colors[0].index(cr)]])

    return colorlookup, colors, edges

def funcCompress3v(x,y,z,func,dir="bi",algo="greedy"): #max 3 inputs
    number_of_var = 3
    colorlookup = []
    combination = []
    g1 = []
    entropy =[]
    edges = []
    Cr_num = []
    colors = []
    length = [len(x),len(y),len(z)]
    temp_loop = [[0,1,2],[1,0,2],[2,0,1]]

    for i in range(number_of_var):
        temp_combination = [[[] for c in range(length[temp_loop[i][1]])] for gc in range(length[temp_loop[i][0]])]
        temp_g1 = [[] for gc in range(length[i])]
        temp_edge = []

        for p in range(length[temp_loop[i][0]]):  # table for possible valve values.
            for e in range(length[temp_loop[i][1]]):
                for d in range(length[temp_loop[i][2]]):
                    if i==0:
                        v = func(x[p], y[e], z[d])  # for each x, all y and for each y, all z
                    if i==1:
                        v = func(x[e], y[p], z[d])  # for each y, all x and for each x, all z
                    if i==2:
                        v = func(x[e], y[d], z[p])  # for each z, all y and for each y, all x
                    temp_combination[p][e].append(v)
        # print(valvelookup_err)

        for gc in range(length[i]):
            for jc in range(length[i]):
                if temp_combination[gc] == temp_combination[jc]:
                    continue
                else:
                    if dir == "uni":
                        temp_g1 = uniaddEdge(temp_g1, gc, jc)
                    elif dir == "bi":
                        temp_g1 = diraddEdge(temp_g1, gc, jc)
                    Edge(temp_edge, gc, jc)
        # print(g1)
        if algo == "hierarchy":
            temp_Cr_num, temp_colors = hierarchyColoring(temp_g1, length[i])  # graph coloring
        elif algo == "greedy":
            temp_Cr_num, temp_colors = greedyColoring(temp_g1, length[i])  # graph coloring
        temp_entropy = entropy_count(temp_Cr_num, temp_colors)
        print("chromatic number for var ",i,": ", temp_Cr_num, temp_colors, temp_entropy)
        combination.append(temp_combination)
        entropy.append(temp_entropy)
        g1.append(temp_g1)
        edges.append(temp_edge)
        Cr_num.append(temp_Cr_num)
        colors.append(temp_colors)

    #return Cr_num, colors, combination, g1Edge


    for xcr in range(Cr_num[0]):
        for ycr in range(Cr_num[1]):
            for zcr in range(Cr_num[2]):
                colorlookup.append([xcr, ycr, zcr, combination[0][colors[0].index(xcr)][colors[1].index(ycr)][colors[2].index(zcr)]])

    return colorlookup, colors, edges, entropy

def funcCompressPC(bound_min,bound_max,grid_step):
    grid_space = np.mgrid[bound_min:(bound_max+grid_step):grid_step, bound_min:(bound_max+grid_step):grid_step, bound_min:(bound_max+grid_step):grid_step].reshape(3, -1).T
    print(len(grid_space))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_space)
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(len(np.asarray(pcd.points)), 3)))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.04)
    o3d.visualization.draw_geometries([voxel_grid])
    #prinp.asarray(pcd.points)
    #voxel_len = len(voxel_grid.get_voxels()) #colorlookup

    #v = np.asarray([voxel_grid.get_voxel(pt) for pt in queries])
    #entropy = entropy_count(voxel_len,v.tolist())
    colorlookup = np.asarray([pt.grid_index for pt in voxel_grid.get_voxels()])
    colorlookup= colorlookup.tolist()
    colorlookup.sort()

    print("Number of Color: ",len(colorlookup))

    return colorlookup, voxel_grid, len(grid_space)




