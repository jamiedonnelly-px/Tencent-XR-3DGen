'''
 # @ Copyright: Copyright 2022 Tencent Inc
 # @ Author: shenzhou
 # @ Create Time: 1970-01-01 08:00:00
 # @ Description:
 '''

import os
import argparse
import multiprocessing as mp

def run_command(cmd):
    print('cmd', cmd)
    os.system(cmd)
    
def run_commands_in_parallel(commands, pool_count=-1):
    commands = [cmd for cmd in commands if cmd]
    if not commands:
        print('[ERROR] invalid commands')
        return
    
    pool_count = os.cpu_count() if pool_count < 0 else pool_count
    print(f'run {len(commands)} cmds by {pool_count} pool')
    with mp.Pool(processes=pool_count) as pool:
        async_results = [pool.apply_async(run_command, (cmd,)) for cmd in commands]
        results = [r.get() for r in async_results]
        
        # for r in async_results:
        #     r.wait()
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='triplane')
    parser.add_argument('cmds_txt', type=str, help='each line is a cmd')
    parser.add_argument('--pool_count', type=int, default=-1)
    args = parser.parse_args()
    
    with open(args.cmds_txt, 'r') as f:
        commands = [line.strip() for line in f.readlines()]    
        run_commands_in_parallel(commands, pool_count=args.pool_count)
