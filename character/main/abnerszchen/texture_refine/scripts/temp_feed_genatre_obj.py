import os
import json
import argparse

def feed_generate_obj(in_json, in_objs_txt, out_json):

    with open(in_json, encoding='utf-8') as f:
        raw_dict = json.load(f)    
        
    obj_dict = {}
    lines = [line.strip() for line in open(in_objs_txt, "r").readlines()]
    for line in lines:
        oname = line.split('/')[-2]
        obj_path = os.path.join(line, 'mesh.obj')
        if not os.path.exists(obj_path):
            continue        
        obj_dict[oname] = obj_path
        
    data_dict = raw_dict['data']
    new_data_dict = {}
    for dname, onames_dict in data_dict.items():
        for oname, meta in data_dict[dname].items():
            if oname not in obj_dict:
                continue
            
            if dname not in new_data_dict:
                new_data_dict[dname] = {}
            
            new_data_dict[dname][oname] = meta
            new_data_dict[dname][oname]['diffusion_obj'] = obj_dict[oname]
            
            
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as jf:
        jf.write(json.dumps({'data': new_data_dict}, indent=4))          
    print(f'make raw {in_json} to depth {out_json}')    
    
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='feed diffusion_obj')
    parser.add_argument('in_json', type=str)
    parser.add_argument('in_objs_txt', type=str)
    parser.add_argument('out_json', type=str)
    args = parser.parse_args()
    
    feed_generate_obj(args.in_json, args.in_objs_txt, args.out_json)


if __name__ == "__main__":
    main()
