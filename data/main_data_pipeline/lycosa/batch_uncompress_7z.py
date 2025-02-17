import argparse
import os
import shutil
import time


def skip_names_in_set(skip_set, name):
    for skip_name in skip_set:
        if skip_name in name:
            return True
    return False


def organize_files(filename_list: list):
    filename_list.sort()
    filename_set = set(filename_list)
    basename_path_map = {}
    basename_part_map = {}

    for filename in filename_set:
        file_basename = os.path.splitext(filename)[0]
        file_extension = os.path.splitext(filename)[1]
        if file_extension.lower() in [".rar", ".zip", ".tar"]:
            if file_extension != ".rar":
                if file_basename in basename_path_map:
                    basename_path_map[file_basename].append(filename)
                else:
                    basename_path_map[file_basename] = []
                    basename_path_map[file_basename].append(filename)
            else:
                if ".part" in file_basename:
                    real_basename_index = file_basename.find(".part")
                    real_basename = file_basename[0:real_basename_index]
                else:
                    real_basename = file_basename

                if real_basename not in basename_path_map:
                    basename_path_map[real_basename] = []
                    basename_path_map[real_basename].append(filename)
                else:
                    basename_path_map[real_basename].append(filename)

                # print(file_basename,(".part01." in file_basename))
                # print(file_basename,(".part1." in file_basename))
                if ".part01." in filename or ".part1." in filename:
                    basename_part_map[real_basename] = filename

    print(basename_path_map)
    print(basename_part_map)
    return basename_path_map, basename_part_map


def apply_decompress_folder(folder_path: str):
    compress_folder = folder_path
    files = os.listdir(compress_folder)
    basename_filename_map, basename_part_map = organize_files(files)

    decompressed_folder_list = []

    for basename in basename_filename_map.keys():
        new_dir = os.path.join(compress_folder, basename)
        if not os.path.exists(new_dir):
            try:
                os.mkdir(new_dir)
            except:
                print("Found special charactors that is not supported by current system; skip %s" % (
                    basename_filename_map[basename]))
                continue
        for filename in basename_filename_map[basename]:
            file_fullpath = os.path.join(compress_folder, filename)
            new_file_fullpath = os.path.join(new_dir, filename)
            shutil.move(file_fullpath, new_file_fullpath)
            time.sleep(0.1)

        if basename in basename_part_map.keys():
            real_to_extract = os.path.join(
                new_dir, basename_part_map[basename])
            extract_dir = os.path.join(new_dir, "extract")
            if not os.path.exists(extract_dir):
                os.mkdir(extract_dir)
            unrar_cmd = "7z x \'{}\' -o\'{}\'".format(
                real_to_extract, extract_dir)

            print(unrar_cmd)
            os.system(unrar_cmd)
            time.sleep(0.1)
            decompressed_folder_list.append(extract_dir)
        else:
            for filename in basename_filename_map[basename]:
                real_to_extract = os.path.join(new_dir, filename)
                file_extension = os.path.splitext(filename)[1]
                lower_file_extension = file_extension.lower()
                if lower_file_extension == ".zip":
                    extract_dir = os.path.join(new_dir, "extract")
                    if not os.path.exists(extract_dir):
                        os.mkdir(extract_dir)

                    unzip_cmd = "7z x  \'{}\' -o\'{}\'".format(
                        new_file_fullpath, extract_dir)
                    print(unzip_cmd)
                    os.system(unzip_cmd)
                    time.sleep(0.1)
                    decompressed_folder_list.append(extract_dir)
                elif lower_file_extension == ".rar":
                    extract_dir = os.path.join(new_dir, "extract")
                    if not os.path.exists(extract_dir):
                        os.mkdir(extract_dir)

                    unzip_cmd = "7z x \'{}\' -o\'{}\'".format(
                        new_file_fullpath, extract_dir)
                    print(unzip_cmd)
                    os.system(unzip_cmd)
                    time.sleep(0.1)
                    decompressed_folder_list.append(extract_dir)
                # elif lower_file_extension == ".tar":
                #     extract_dir = os.path.join(new_dir, "extract")
                #     if not os.path.exists(extract_dir):
                #         os.mkdir(extract_dir)

                #     untar_cmd = "tar -xf \'{}\' --directory \'{}\'  > /dev/null".format(
                #         new_file_fullpath, extract_dir)
                #     print(untar_cmd)
                #     os.system(untar_cmd)
                #     time.sleep(0.1)
                #     decompressed_folder_list.append(extract_dir)

    return decompressed_folder_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate mesh list from a group of organized mesh folder')
    parser.add_argument('--decompress_folder', nargs='+',
                        help='mesh folder to decompress')
    parser.add_argument('--pool_cnt', type=int, default=24,
                        help='multiprocessing pool cnt')
    args = parser.parse_args()

    decompress_folder_list = args.decompress_folder
    pool_cnt = args.pool_cnt

    # pool = ThreadPoolExecutor(max_workers=pool_cnt,
    #                           thread_name_prefix='decompress')

    # decompress_folder_list = [
    #     "/media/steve/760474EB0474B02B/Models/www.cgmodel.com-20230825/"]
    future_decompress_candidates = []
    for folder_name in decompress_folder_list:
        new_decompress_folder_list = apply_decompress_folder(folder_name)
        future_decompress_candidates = new_decompress_folder_list + future_decompress_candidates

    for folder_name in future_decompress_candidates:
        apply_decompress_folder(folder_name)
        time.sleep(0.1)

    # pool.shutdown()
    # time.sleep(0.1)
