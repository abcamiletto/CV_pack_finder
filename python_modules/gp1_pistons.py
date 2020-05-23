import math

def point_distance(x_1,y_1, x_2,y_2):
    dist_vector = [ x_1 - x_2 , y_1 - y_2 ]
    dist_vec_module = math.sqrt(dist_vector[0] ** 2 + dist_vector[1] ** 2  )
    return dist_vec_module
def nearest_pist(x,y, pistons_data):
    n_pist = 100
    best_dist = 100
    output = [None] * 2
    for idx, piston in enumerate(pistons_data,1) :
        if x < piston[0] - 0.02:
            if x < piston[0]:
                temp_res = point_distance(x, y, piston[0], piston[1])
                if temp_res < best_dist:
                    best_dist = temp_res
                    n_pist = idx

    output[0] = n_pist
    output[1] = best_dist
    return output
def piston_list_generator(packs_data, pistons_data, th):
    pack_info = []
    for pack in packs_data: #Storing info about all the packs
        pack_result = nearest_pist(pack[0][0], pack[0][1], pistons_data)
        pack_info.append(pack_result)

    #selecting only pack nearby pistons
    piston_selected = [x[0] for x in pack_info if (x[1] < th)]

    #deleting Duplicates
    output = list(set(piston_selected))

    return output

def new_nearest_pist(x,y, pistons_data, piston_threshold):
    n_pist = 100
    best_dist = 100
    output = [None] * 3
    for idx, piston in enumerate(pistons_data,1) :
        #FIRST PART: BACKWARD CONTROL
        if idx > piston_threshold:
            if x > piston[0]:
                temp_res = point_distance(x, y, piston[0], piston[1])
                if temp_res < best_dist :
                    best_dist = temp_res
                    n_pist = idx
        #SECOND PART: FORWARD CONTROL
        else:
            if x < piston[0] :
                if x < piston[0] :
                    temp_res = point_distance(x, y, piston[0], piston[1])
                    if temp_res < best_dist:
                        best_dist = temp_res
                        n_pist = idx

    output[0] = n_pist
    output[1] = best_dist
    output[2] = [x,y]
    return output
def are_there_collision(pack, passed_packs):
    for passed_pack in passed_packs:
        #if point_distance_list(passed_pack[2], pack[2]) < passed_pack[3]/2 + pack[3]/2 + 0.3: return True
        if passed_pack[2][0] - pack[2][0] < 50: return True
    return False
def new_piston_list_generator(pack_data, pistons_data, piston_threshold, th):
    #Storing valuable information
    x_back_pist = pistons_data[0][0]
    x_threshold_pist = pistons_data[piston_threshold][0]
    pack_info = []

    for idx, pack in enumerate(pack_data): #Storing info about all the packs
        pack_result = new_nearest_pist(pack[0][0], pack[0][1], pistons_data, piston_threshold)
        pack_info.append(pack_result)

    pack_sorted = sorted(pack_info, key=lambda item: item[2][0], reverse=True)

    packs_to_ignore = -1
    for idx, pack in enumerate(pack_sorted):
        if pack[2][0] > x_back_pist:
            continue
        if pack[2][0] > x_threshold_pist:
            if idx == 0: packs_to_ignore = 0
            else:
                passed_packs = pack_sorted[:idx]
                if are_there_collision(pack, passed_packs):
                    pass
                    break
                else:
                    packs_to_ignore = idx
        else:
            pass
            break

    if pack_sorted[-1][2][0] > x_back_pist : packs_to_ignore = len(pack_sorted) - 1

    pack_selected = pack_sorted[packs_to_ignore+1:]
    #selecting only pack nearby pistons
    piston_selected = [x[0] for x in pack_selected if (x[1] < th)]

    #deleting Duplicates
    output = list(set(piston_selected))

    return output

if __name__ == "__main__":
    pistons_data = [[11123, 675], [10022, 100], [7780, 45], [7000, 445], [2780, 415] , [752, 1205], [78, 145], [15, 99], ]
    packs_data = [[[521, 101], 35, 'time'],
                  [[1521, 31], 35, 'time'],
                  [[5721, 3401], 35, 'time'],
                  [[7520, 1201], 35, 'time'],
                  [[11121, 660], 35, 'time']
                  ]

    print(piston_list_generator(packs_data, pistons_data, 100))

    print(new_piston_list_generator(packs_data, pistons_data, 3, 100))
