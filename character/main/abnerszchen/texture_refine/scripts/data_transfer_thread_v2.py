import os
import threading
import queue
import time
import argparse

all_file_num = 0
copy_ok_num = 0

class CopyWorker(threading.Thread):
    def __init__(self, queue, dest_folder, lock):
        super().__init__()
        self.queue = queue
        self.dest_folder = dest_folder
        self.lock = lock

    def run(self):
        global copy_ok_num
        while True:
            files_path = self.queue.get()
            if files_path is None:
                self.queue.task_done()
                break
            src_file, dst_file = files_path
            if not os.path.exists(dst_file):
                # os.system("rsync -a '{}' '{}'".format(src_file, dst_file))
                os.system("rsync '{}' '{}'".format(src_file, dst_file))
            if os.path.islink(src_file):
                os.system("ln -s '{}' '{}'".format(src_file, dst_file))

            with self.lock:
                copy_ok_num += 1
                print("\rtransfer percentage: %.2f %%" % (copy_ok_num*100/all_file_num), 'transfer sample amount: {}/{}'.format(copy_ok_num, all_file_num), end="", flush=True)

            self.queue.task_done()


class TraverseWorker(threading.Thread):
    def __init__(self, dir_queue, files, lock):
        super().__init__()
        self.queue = dir_queue
        self.lock = lock
        self.files = files

    def run(self):
        while True:
            dir_paths = self.queue.get()
            if dir_paths is None:
                self.queue.task_done()
                break
            src_dir, dst_dir = dir_paths
            get_transfer_list(src_dir, dst_dir, self.files)
            self.queue.task_done()

def get_transfer_dir(src_dir, dst_dir, dir_queue):
    try:
        for subfile in os.listdir(src_dir):
            file = os.path.join(src_dir, subfile)
            if os.path.isdir(file):
                s_dir = os.path.join(src_dir, subfile)
                d_dir = os.path.join(dst_dir, subfile)
                dir_queue.put([s_dir, d_dir])
                get_transfer_dir(s_dir, d_dir, dir_queue)

    except Exception as e:
        print(e)
        return


def get_transfer_list(src_dir, dst_dir, files):
    global all_file_num
    try:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        for subfile in os.listdir(src_dir):
            file = os.path.join(src_dir, subfile)
            if os.path.isfile(file):
                    src_file_path = os.path.join(src_dir, subfile)
                    dst_file_path = os.path.join(dst_dir, subfile)
                    files.put([src_file_path, dst_file_path])
                    all_file_num += 1
    except Exception as e:
        print(e)
        return


def copy_files(src_folder, dest_folder, num_workers=5):
    dir_queue = queue.Queue()
    files = queue.Queue()
    lock = threading.Lock()

    traverse_workers = []
    for _ in range(num_workers // 2):
        worker = TraverseWorker(dir_queue, files, lock)
        worker.start()
        traverse_workers.append(worker)

    copy_workers = []
    for _ in range(num_workers // 2):
        worker = CopyWorker(files, dest_folder, lock)
        worker.start()
        copy_workers.append(worker)


    get_transfer_dir(src_folder, dest_folder, dir_queue)
    # get_transfer_list(src_folder, dest_folder, files)



    for _ in range(num_workers // 2):
        dir_queue.put(None)
        files.put(None)

    dir_queue.join()

    for worker in traverse_workers:
        worker.join()
    for worker in copy_workers:
        worker.join()

    new_copy_workers = []
    while True:
        flag_all = False
        for worker in traverse_workers:
            if not worker.is_alive():
                worker = CopyWorker(files, dest_folder, lock)
                worker.start()
                new_copy_workers.append(worker)
            else:
                flag_all = False
        time.sleep(10)
        if flag_all:
            break
    for _ in range(num_workers // 2):
        files.put(None)
    files.join()
    for worker in new_copy_workers:
        worker.join()

    print("Traverse worker stoped!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('src_dir', type=str)
    parser.add_argument('dst_dir', type=str)
    args = parser.parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    
    start_time = time.time()
    # src_dir = "/apdcephfs/private_neoshang/code"
    # dst_dir = "/aigc_cfs_2/neoshang/code"
    thread_num = 88
    
    copy_files(src_dir, dst_dir, thread_num)

    end_time = time.time()
    elapsed_time = end_time - start_time  # 计算程序运行时间（秒）

    # 转换为时分秒
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'程序运行时间: {int(hours)}:{int(minutes)}:{seconds:.2f}')