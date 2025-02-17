import copy
import os
import glob
import pathlib
from pathlib import Path
import json

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data

def write_json(fname , data  ):
    json_object = json.dumps(data, indent=4)
    with open(fname, "w") as outfile:
        outfile.write(json_object)

def pprint(j):
    j = json.dumps(j, indent=2)
    print( j )




class LayeredJson:
    def __init__(self, fpath):
        self.object = load_json( fpath )
    def show_cls(self):
         print( "classes:", self.object["data"].keys() )
    def separate_by_cls(self):
        pass
        # for cls in clses:
        #     screenshot_path = os.path.join(screenshot_root, cls)
        #     Path(screenshot_path).mkdir(exist_ok=True)

    def count_cls(self, clses=None):
        cnt = 0
        cnt_none = 0

        if clses is None:
            clses = self.object["data"].keys()

        for cls in clses:
            ids = self.object["data"][cls].keys()
            cnt += len ( ids )
            for id in ids:
                if self.object["data"][cls][id]["body_key"] is None:
                    cnt_none +=1
                    # print( cls, id)

        print("cls assets in", clses, "\n ----total #:", cnt, "No body-key #:" , cnt_none)


    def remove_keys ( self, clses, keys ):

        duplicate = copy.deepcopy( self.object)

        cnt = 0
        for cls in clses:
            ids = self.object["data"][cls].keys()
            for id in ids:
                if id in keys :
                    # print( cls, id )
                    del duplicate["data"][cls][id]
                    cnt +=1

        print( "remove ", cnt, "keys")

        self.object = duplicate


    def replace_keys (self,  clses, keys, source ) :

        duplicate = copy.deepcopy( self.object)

        cnt = 0

        for cls in clses:
            ids = self.object["data"][cls].keys()
            for id in ids:
                if id in keys :
                    # print( cls, id )
                    source[id]["Gender"] = duplicate["data"][cls][id]["Gender"]
                    duplicate["data"][cls][id] = source[id]
                    cnt +=1


        print( "replace ", cnt, "keys")


        self.object = duplicate

    def write_unified_bodykey(self, clses, unified_bodykey):

        for cls in clses:
            ids = self.object["data"][cls].keys()
            for id in ids:
                if self.object["data"][cls][id]["body_key"] is None:
                    self.object["data"][cls][id]["body_key"] = unified_bodykey
                else :
                    print( "--------------body key exist---------------", id, self.object["data"][cls][id]["body_key"])



    def write_bodykey_mcwy1(self, ):

        clses = ['MCWY_1_Top', 'MCWY_1_Outfit', 'MCWY_1_Bottom', 'MCWY_1_Shoe']

        for cls in clses:
            ids = self.object["data"][cls].keys()
            for id in ids:
                if self.object["data"][cls][id]["body_key"] is None:
                    gender = self.object["data"][cls][id]["Gender"]
                    if gender == "Male" :
                        self.object["data"][cls][id]["body_key"] = [ "male", "mcwy1_old"]
                    elif gender == "Female":
                        self.object["data"][cls][id]["body_key"] = ["female", "mcwy1_old"]
                    else:
                        print ( "---gender undefined--")
                else:
                    print( "--------------body key exist---------------", id, self.object["data"][cls][id]["body_key"])


    def split_json_via_body_key_existence(self):

        complete_part = {"data": {} }
        incomplete_part = {"data": {} }

        clses = self.object["data"].keys()

        for cls in clses:
            complete_part["data"][cls] = {}
            incomplete_part["data"][cls] = {}

            ids = self.object["data"][cls].keys()
            for id in ids:
                if self.object["data"][cls][id]["body_key"] is None:
                    incomplete_part ["data"][cls][id] = self.object["data"][cls][id]
                else:
                    complete_part ["data"][cls][id] = self.object["data"][cls][id]

        return complete_part, incomplete_part


    # def delete_keys(self, keys):
    #     for key in keys:
    #         del self.object [key]

    def integrate_json(self, new_json, clses ):


        for cls in clses:
            ids = new_json.object["data"][cls].keys()
            for id in ids:
                self.object["data"][cls][id] = new_json.object["data"][cls][id]

        for cls in clses:
            new_json.object["data"][cls] = {}

        return self, new_json


    def split_json_for_jobs(self, clses, nsplits, dunmp_folder , shuffle=True):

        def divide_chunks(l, n):
            # looping till length l
            for i in range(0, len(l), n):
                yield l[i:i + n]

        class_keys = []

        for cls in clses:
            ids = self.object["data"][cls].keys()
            for id in ids:
                class_keys.append( [cls, id] )

        n_ids = len(class_keys)



        if shuffle :
            import random
            random.shuffle(class_keys)

        n = n_ids // nsplits + 1
        list_clskey = list(divide_chunks( class_keys, n ))


        pathlib.Path(dunmp_folder).mkdir(exist_ok=True, parents=True)

        for idx, sublst in enumerate( list_clskey ):
            subjson = {"data": {}}
            for cls in clses:
                subjson["data"][cls] = {}
            for ele in sublst:
                subjson["data"][ele[0]][ele[1]] = self.object["data"][ele[0]][ele[1]]

            write_json( os.path.join( dunmp_folder, "part%03d"%idx + ".json"), subjson)







def separate_ruku_json():

    all = "./jsons/body_key_all_vroid_correct.json"
    lj = LayeredJson(all)
    # lj.show_cls()



    all_cls = ['MCWY_1_Top',
               'MCWY_1_Outfit',
               'MCWY_1_Bottom',
               'MCWY_1_Shoe',
               'readyplayerme_top',
               'readyplayerme_hair',
               'readyplayerme_footwear',
               'readyplayerme_hat',
               'readyplayerme_bottom',
               'MCWY_2_Hair',
               'MCWY_2_Glove',
               'MCWY_2_Sock',
               'MCWY_2_Bottom',
               'MCWY_2_Dress',
               'MCWY_2_Shoe',
               'MCWY_2_Top',
               'VRoid_VRoid_Top',
               'VRoid_VRoid_Bottom',
               'VRoid_VRoid_Hair',
               'VRoid_VRoid_Shoe',
               'DAZ_DAZ_Bottom',
               'DAZ_DAZ_Outfit',
               'DAZ_DAZ_Shoe',
               'DAZ_DAZ_Top']

    daz = ['DAZ_DAZ_Bottom', 'DAZ_DAZ_Outfit', 'DAZ_DAZ_Shoe', 'DAZ_DAZ_Top']
    mcwy2 = [ 'MCWY_2_Hair', 'MCWY_2_Glove', 'MCWY_2_Sock', 'MCWY_2_Bottom', 'MCWY_2_Dress', 'MCWY_2_Shoe', 'MCWY_2_Top' ]
    readypm =[ 'readyplayerme_top', 'readyplayerme_hair', 'readyplayerme_footwear', 'readyplayerme_hat', 'readyplayerme_bottom']
    mcwy1 = ['MCWY_1_Top', 'MCWY_1_Outfit', 'MCWY_1_Bottom', 'MCWY_1_Shoe']
    vroid = [ 'VRoid_VRoid_Top', 'VRoid_VRoid_Bottom', 'VRoid_VRoid_Hair', 'VRoid_VRoid_Shoe' ]

    # body key check
    lj.count_cls(daz)
    lj.count_cls(mcwy2)
    lj.count_cls(mcwy1)
    lj.count_cls(readypm)
    lj.count_cls(vroid)

    # write body key for daz
    lj.write_unified_bodykey( clses=daz,  unified_bodykey = ["male", "daz"] )



    # write body key for mcwy1
    # lj.write_bodykey_mcwy1()


    # body key check
    print("------------------------------------------------------------------------------------------------------------")
    lj.count_cls(daz)
    lj.count_cls(mcwy2)
    lj.count_cls(mcwy1)
    lj.count_cls(readypm)
    lj.count_cls(vroid)


    # # separate json
    complete_json, incomplete_json = lj.split_json_via_body_key_existence()
    write_json( "./jsons/20240704_ruku_ok.json" , complete_json)
    write_json( "./jsons/20240704_ruku_not_ok.json", incomplete_json)



def split_json_for_registration():

    ruku_not_ok = "./jsons/20240704_ruku_not_ok.json"
    lj = LayeredJson(ruku_not_ok)

    vroid = [ 'VRoid_VRoid_Top', 'VRoid_VRoid_Bottom', 'VRoid_VRoid_Hair', 'VRoid_VRoid_Shoe' ]


    lj.split_json_for_jobs(vroid, 8, "./jsons/job_jsons/")


def ruku():

    no_ok_json = "./jsons/20240708_ruku_not_ok.json"
    no_ok_json = LayeredJson(no_ok_json)


    ruku_ok_json = "./jsons/20240704_ruku_ok.json"
    ruku_ok_json = LayeredJson(ruku_ok_json)


    daz = ['DAZ_DAZ_Bottom', 'DAZ_DAZ_Outfit', 'DAZ_DAZ_Shoe', 'DAZ_DAZ_Top']
    mcwy2 = [ 'MCWY_2_Hair', 'MCWY_2_Glove', 'MCWY_2_Sock', 'MCWY_2_Bottom', 'MCWY_2_Dress', 'MCWY_2_Shoe', 'MCWY_2_Top' ]
    readypm =[ 'readyplayerme_top', 'readyplayerme_hair', 'readyplayerme_footwear', 'readyplayerme_hat', 'readyplayerme_bottom']
    mcwy1 = ['MCWY_1_Top', 'MCWY_1_Outfit', 'MCWY_1_Bottom', 'MCWY_1_Shoe']
    vroid = [ 'VRoid_VRoid_Top', 'VRoid_VRoid_Bottom', 'VRoid_VRoid_Hair', 'VRoid_VRoid_Shoe' ]

    # body key check
    ruku_ok_json.count_cls(vroid)
    no_ok_json.count_cls(vroid)


    # write body key for vroid
    no_ok_json.write_unified_bodykey( clses=vroid,  unified_bodykey = ["male", "vroid"] )



    #integrate json
    ruku_ok_json, no_ok_json = ruku_ok_json.integrate_json( no_ok_json, vroid )
    ruku_ok_json.count_cls(vroid)
    no_ok_json.count_cls(vroid)


    # dump json
    write_json( "./jsons/20240709_ruku_ok.json" , ruku_ok_json.object)
    write_json( "./jsons/20240709_ruku_not_ok.json", no_ok_json.object)


    # # write body key for mcwy1
    # # lj.write_bodykey_mcwy1()
    #
    #
    # # body key check
    # print("------------------------------------------------------------------------------------------------------------")
    # lj.count_cls(daz)
    # lj.count_cls(mcwy2)
    # lj.count_cls(mcwy1)
    # lj.count_cls(readypm)
    # lj.count_cls(vroid)


def fix_mcwy_A2T():

    old_gdp = "./jsons/web_flatten_gdp.json"
    ruku_ok = "./jsons/20240710_ruku_ok.json"
    ruku_ok_gdp = "./jsons/20240710_ruku_ok_gdp.json"
    ruku_not_ok = "./jsons/20240710_ruku_not_ok.json"

    old_gdp = LayeredJson(old_gdp)
    ruku_ok = LayeredJson(ruku_ok)
    ruku_ok_gdp = LayeredJson(ruku_ok_gdp)
    ruku_not_ok = LayeredJson(ruku_not_ok)

    filtered = glob.glob( os.path.join( "/home/rabbityl/tboard/mcwy_ss/*.jpg") )
    filtered = [ e.split("/")[-1][:-4] for e in filtered ]

    # print( filtered )

    replace = []
    remove = []

    old_gdp_filter =  {}

    for k in old_gdp.object.keys():
        if old_gdp.object[k]["body_key"][1] == "mcwy2_A":
            if k in filtered:
                replace.append(k)
                old_gdp_filter[k] = old_gdp.object[k]
            else:
                remove.append(k)

    # print( len( replace ))
    # print( len( remove ))

    mcwy2 = [ 'MCWY_2_Hair', 'MCWY_2_Glove', 'MCWY_2_Sock', 'MCWY_2_Bottom', 'MCWY_2_Dress', 'MCWY_2_Shoe', 'MCWY_2_Top' ]

    # remove no good keys
    ruku_ok.remove_keys( mcwy2, remove)
    ruku_ok_gdp.remove_keys(mcwy2, remove)

    # replace keys
    ruku_ok.replace_keys ( mcwy2, replace, old_gdp_filter )
    ruku_ok_gdp.replace_keys ( mcwy2, replace, old_gdp_filter )

    # dump json
    write_json("./jsons/20240711_ruku_ok.json", ruku_ok.object)
    write_json("./jsons/20240711_ruku_ok_gdp.json", ruku_ok_gdp.object)

    # ruku_ok.remove_keys(  mcwy2, remove)


def extract_key_smpl_pair(   ):


    ruku_ok = "./jsons/20240711_ruku_ok_gdp.json"

    ruku_ok = LayeredJson(ruku_ok)


    clses = ['VRoid_VRoid_Top', 'VRoid_VRoid_Bottom', 'VRoid_VRoid_Hair', 'VRoid_VRoid_Shoe']

    key_smpl_map = {}

    for cls in clses:
        ids = ruku_ok.object["data"][cls].keys()
        for id in ids  :
            mesh = ruku_ok.object["data"][cls][id]["Mesh"]
            key_smpl_map[ id ] = os.path.join( pathlib.Path(mesh).parent.parent, "smplx_and_offset_smplified.npz" )

    write_json("./jsons/20240711_key_smpl_map.json", key_smpl_map)




if __name__ == '__main__':


    # split_json_for_registration()
    # ruku()


    # fix_mcwy_A2T()


    # extract

    extract_key_smpl_pair( )