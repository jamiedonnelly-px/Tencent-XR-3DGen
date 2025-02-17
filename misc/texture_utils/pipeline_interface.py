import os
import pulsar
from _pulsar import ConsumerType, LoggerLevel
import argparse
import json
import time
import uuid
import logging
import shutil
import traceback
import sys
import numpy as np

codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)

from cos_api import CosClient
from interfaces_3d import Text2ImgInterface, T2MVDInterface, T2iMmdInterface, Zero123plusInterface, ImageDreamInterface, MmdInterface
from interfaces_3d import LrmInterface, CraftsManInterface, MV2MeshInterface, ClayInterface
from interfaces_3d import Mesh2ImageInterface, ConsistentTexInterface, D2rgbInterface, ImgSRInterface
from interfaces_3d import (BakingInterface, Verts2TexInterface, TextureBakingInterface, TexBakeInpaintInterface,
                           SyncMVDInterface, SDXLInpaintInterface, TexallInterface, Texall2Interface)
from interfaces_3d import QuadRemeshInterface, BlenderCvtInterface, GpuToolInterface, init_job_id
from task_queue import TaskQueue
from remote_api import determine_building_img

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def parse_relative_path(file_path, root_dir):
    if os.path.exists(file_path):
        abs_path = os.path.abspath(file_path)
    else:
        abs_path = os.path.join(root_dir, file_path)
    assert os.path.exists(abs_path), f"can not parse_relative_path {file_path} and {root_dir}"
    return abs_path


class Generate3DInterface():

    def __init__(
        self,
        cfg_json="configs/client_generate_3d.json",
    ):
        # TODO read from cfg
        try:
            logging.info(f"load Generate3DInterface cfg from: {cfg_json}")
            self.cfg_json, self.cfg_dict = self.load_cfg(cfg_json)
            self.pipe_dict = self.cfg_dict["pipe_dict"]
            self.open_task_queue = self.cfg_dict.get("open_task_queue", False)
            self.task_query_interval = self.cfg_dict.get("task_query_interval", 2)
            
            self.cos_client = CosClient(parse_relative_path("configs/cos_cfg.json", codedir))

            # step1
            self.text2img_interface = Text2ImgInterface(parse_relative_path(self.cfg_dict["text2img"]["cfg_json"], codedir))
            self.t2mvd_interface = T2MVDInterface(parse_relative_path(self.cfg_dict["t2mvd"]["cfg_json"], codedir))
            self.t2i_mmd_interface = T2iMmdInterface(parse_relative_path(self.cfg_dict["t2i_mmd"]["cfg_json"], codedir))
            self.zero123plus_interface = Zero123plusInterface(
                parse_relative_path(self.cfg_dict["zero123plus"]["cfg_json"], codedir))
            self.imagedream_interface = ImageDreamInterface(
                parse_relative_path(self.cfg_dict["imagedream"]["cfg_json"], codedir))
            self.mmd_interface = MmdInterface(parse_relative_path(self.cfg_dict["mmd"]["cfg_json"], codedir))

            # step2
            self.lrm_interface = LrmInterface(parse_relative_path(self.cfg_dict["lrm"]["cfg_json"], codedir))
            self.crman_interface = CraftsManInterface()
            self.mv2mesh_interface = MV2MeshInterface(parse_relative_path(self.cfg_dict["mv2mesh"]["cfg_json"], codedir))
            self.clay_interface = ClayInterface(parse_relative_path(self.cfg_dict["clay"]["cfg_json"], codedir))
            self.clay_1img_interface = ClayInterface(parse_relative_path(self.cfg_dict["clay_1img"]["cfg_json"], codedir))

            # step3
            self.mesh2image_interface = Mesh2ImageInterface(parse_relative_path(self.cfg_dict["mesh2image"]["cfg_json"], codedir))
            self.consistent_tex_interface = ConsistentTexInterface(parse_relative_path(self.cfg_dict["consistent_tex"]["cfg_json"], codedir))
            self.d2rgb_interface = D2rgbInterface(parse_relative_path(self.cfg_dict["d2rgb"]["cfg_json"], codedir))
            self.imgsr_interface = ImgSRInterface(parse_relative_path(self.cfg_dict["imgsr"]["cfg_json"], codedir))

            # step4
            self.texall_interface = TexallInterface(
                parse_relative_path(self.cfg_dict["texall"]["cfg_json"], codedir))
            self.texall2_interface = Texall2Interface(
                parse_relative_path(self.cfg_dict["texall2"]["cfg_json"], codedir))
            self.texbakeinpaint_interface = TexBakeInpaintInterface(
                parse_relative_path(self.cfg_dict["texbakeinpaint"]["cfg_json"], codedir))
            self.fast_baking_interface = BakingInterface(
                parse_relative_path(self.cfg_dict["baking"]["cfg_json"], codedir))
            self.verts2tex_interface = Verts2TexInterface(
                parse_relative_path(self.cfg_dict["verts2tex"]["cfg_json"], codedir))
            self.sdxl_texrefine_interface = SDXLInpaintInterface(
                parse_relative_path(self.cfg_dict["sdxl_texrefine"]["cfg_json"], codedir))

            self.texture_baking_interface = TextureBakingInterface(
                parse_relative_path(self.cfg_dict["texture_baking"]["cfg_json"], codedir))
            self.texture_syncmvd_interface = SyncMVDInterface(
                parse_relative_path(self.cfg_dict["syncmvd"]["cfg_json"], codedir))

            # step5
            self.quad_remesh_interface = QuadRemeshInterface(
                parse_relative_path(self.cfg_dict["quad_remesh"]["cfg_json"], codedir))
            self.blender_cvt_interface = BlenderCvtInterface(
                parse_relative_path(self.cfg_dict["blender_cvt"]["cfg_json"], codedir))
            self.gputool_interface = GpuToolInterface(parse_relative_path(self.cfg_dict["gputool"]["cfg_json"],
                                                                          codedir))

            # task queue
            if self.open_task_queue:
                self.task_queue_step1 = TaskQueue(self.t2i_mmd_interface.backend_consumer.redis_db, "step1", self.task_query_interval)
                self.task_queue_step2 = TaskQueue(self.t2i_mmd_interface.backend_consumer.redis_db, "step2", self.task_query_interval)
                logging.info(f"open_task_queue done")

            logging.info(f"init sub-interfaces done from {cfg_json}")
        except Exception as e:
            raise ValueError(f"error when init interfaces from {cfg_json}, error:\n{e}")

    def load_cfg(self, cfg_json):
        cfg_json_ = parse_relative_path(cfg_json, codedir)
        assert os.path.exists(cfg_json_), f"can not find valid cfg_json: {cfg_json}, {cfg_json_}"
        with open(cfg_json_, encoding='utf-8') as f:
            cfg_dict = json.load(f)
            print('load cfg_dict=', cfg_dict)
        return cfg_json_, cfg_dict

    def blocking_call_generate_once(self,
                                    job_id,
                                    prompt=None,
                                    in_image_path_list=None,
                                    out_save_dir=None,
                                    gen_mvimg_type="t2i_mmd",
                                    gen_mesh_type="lrm",
                                    texture_type="syncmvd",
                                    use_d2rgb=False,
                                    use_imgsr=True,
                                    use_texrefine=False,
                                    use_output_glb=True,
                                    extra_params={},
                                    timeout=1000):
        """blocking mode, run generate 3d pipeline, 3 steps, 
        synmvd: text mode use ~300s
        imgsr use ~120s
        baking: img mode use ~ 130s, text mode use ~ 180s

        Args:
            job_id(string), uuid
            prompt(string): 输入文字提示词.
            in_image_path_list(list of string): lit of 输入图片绝对路径. 
            out_save_dir(string): 输出文件夹的根目录. 可以指定路径或者None
            gen_mvimg_type(string): 如果用输入文字控制,=t2i_mmd, t2i_z123 or t2mvd, 如果用输入图像控制可以选 "z123" 或 "w3d" 
            gen_mesh_type(string): 多图生Mesh的方法, 可以选 "crman" or "lrm"
            texture_type(string): 纹理生成方式, 可以选 "baking", "syncmvd" 或 "texture_baking" 
            use_d2rgb(bool): 是否d2rgb
            use_imgsr(bool): 是否超分mvimg
            use_texrefine(bool): 是否sdxl_texrefine
            use_output_glb(bool): 是否转成glb输出
            timeout(int): 单个任务的超时时间(s)
        Returns:
            success_flag=T/F, result_mesh=输出mesh路径 , gif_local_path
        """
        try:
            # step1-2. generate coarse mesh(geom only): mv + lrm
            success_flag, result_mesh, gif_local_path = self.main_call_step1_generate_geom_mesh(
                job_id,
                prompt=prompt,
                in_image_path_list=in_image_path_list,
                out_save_dir=out_save_dir,
                gen_mvimg_type=gen_mvimg_type,
                gen_mesh_type=gen_mesh_type,
                use_output_glb=False if texture_type == "baking" else use_output_glb,
                extra_params=extra_params)
            if not success_flag:
                logging.error(
                    f"[ERROR!!!!] main_call_step1_generate_geom_mesh error, input prompt={prompt}; in_image_path_list={in_image_path_list}"
                )
                return False, "", ""

            # step3-5. refine mesh, (mesh2image) (d2rgb), (sr), baking(syncmvd)
            use_mesh2image = False
            if gen_mvimg_type == "t2mvd":
                use_mesh2image = True
            success_flag, result_mesh, gif_local_path = self.main_call_step2_mesh_gen_texture(job_id,
                                                              prompt=prompt,
                                                              gen_mesh_type=gen_mesh_type,
                                                              texture_type=texture_type,
                                                              use_mesh2image=use_mesh2image,
                                                              use_d2rgb=use_d2rgb,
                                                              use_imgsr=use_imgsr,
                                                              use_texrefine=use_texrefine,
                                                              use_output_glb=use_output_glb)

            logging.info(f"[DONE] generate once, success_flag={success_flag}, result_mesh={result_mesh}")
            return success_flag, result_mesh, gif_local_path

        except Exception as e:
            logging.error(f"[ERROR!!!!] blocking_call_generate_once error, input: text:{prompt} /img: {in_image_path_list}, error:\n{e}")
            traceback.print_exc()
            return False, None, None

    def main_call_step1_generate_geom_mesh(
        self,
        job_id,
        prompt=None,
        in_image_path_list=None,
        out_save_dir=None,
        gen_mvimg_type="t2i_mmd",
        gen_mesh_type="lrm",
        use_output_glb=True,
        extra_params={},
    ):
        """blocking call mode, run mv + lrm. generate geom mesh, use ~20s

        Args:
            job_id(string), uuid
            prompt(string): 输入文字提示词.
            in_image_path_list(list of string): list of 输入图片绝对路径. 
            out_save_dir(string): 输出文件夹的根目录. 可以指定路径或者None
            gen_mvimg_type(string): 如果用输入文字控制,=t2i_mmd, t2i_z123 or t2mvd, 如果用输入图像控制可以选 "z123" 或 "w3d" 
            gen_mesh_type(string): 多图生Mesh的方法, 可以选 "crman" or "lrm"
            use_output_glb(bool): 是否转成glb输出
            timeout(int): 单个任务的超时时间(s)
        Returns:
            success_flag=T/F, result_mesh=输出mesh路径, gif_local_path
        """
        try:
            self.update_task_redis(job_id, run_flag="running", step_mode="step1")

            if in_image_path_list is not None:
                logging.info(f"[step0] begin input sr with {in_image_path_list}")
                in_image_path_list = self.imgsr_interface.blocking_call_input_imgsr(job_id + "_inputsr",
                                                                                    in_image_path_list,
                                                                                    os.path.join(out_save_dir, job_id))
                logging.info(f"[step0] end input sr with new: {in_image_path_list}")

            # step1. generate multi-view images
            mv_parameter_dict = self.step1_mv(job_id,
                                              prompt=prompt,
                                              in_image_path_list=in_image_path_list,
                                              out_save_dir=out_save_dir,
                                              gen_mvimg_type=gen_mvimg_type,
                                              extra_params=extra_params)

            # step2. generate mesh (coarse mesh)
            mesh_value_dict = self.step2_mesh(job_id, mv_parameter_dict, gen_mesh_type=gen_mesh_type)
            success_flag = mesh_value_dict["success"]
            result_mesh = os.path.join(mesh_value_dict["result"]["parameter"]["obj_dir"], "obj_mesh_mesh.obj")


            # 6. quad_remesh if not building
            use_quad_remesh = False
            # use_quad_remesh = not determine_building_img(os.path.join(mesh_value_dict["result"]["parameter"]["obj_dir"], "0.png"))
            
            logging.info(f"use_quad_remesh={use_quad_remesh} for job_id = {job_id}")
            if use_quad_remesh and success_flag:
                logging.info(f"begin quad_remesh, job_id = {job_id}")
                run_in_cos, need_run = True, True
                if run_in_cos:
                    quad_in_mesh = os.path.join("/mnt/aigc_bucket_4/pandorax/quad_remesh", job_id,
                                                os.path.basename(result_mesh))
                    job_dir = os.path.dirname(quad_in_mesh)
                    
                    success = self.cos_client.upload_abs_cos_once(result_mesh, quad_in_mesh)
                    if not success:
                        logging.error(f"[Warn] quad run_in_cos but upload_abs_cos_once failed job_id={job_id} ")
                        need_run = False
                else:
                    quad_in_mesh = result_mesh
                    job_dir = os.path.dirname(os.path.dirname(result_mesh))
                if need_run:
                    output_file = os.path.join(job_dir, "quad_remesh/quad_mesh.glb")
                    final_result_dict = self.quad_remesh_interface.blocking_call_quad_remesh(
                        job_id,
                        job_dir,
                        quad_in_mesh,
                        output_file,
                    )
                    quad_flag, result_mesh_quad = self.parse_final_result_dict(
                        job_id, final_result_dict, self.quad_remesh_interface.service_name)
                    if quad_flag:
                        # TODO
                        try:
                            geom_dir = os.path.join(os.path.dirname(os.path.dirname(result_mesh)), "geom")
                            quad_dir = os.path.dirname(result_mesh_quad)
                           
                            cos_abs_path_list, local_path_list = [], []
                            for name in ["quad_mesh.mtl", "kd.png", "quad_mesh.obj"]:
                                cos_abs_path_list.append(os.path.join(quad_dir, name))
                                local_path_list.append(os.path.join(geom_dir, name))
                            quad_flag = self.cos_client.download_abs_cos_threads(cos_abs_path_list, local_path_list)
                        except:
                            quad_flag = False
                            logging.error("warn download_abs_cos_threads failed cos to gdp quad")                            
                            
                    # TODO return cos??
                    result_mesh = result_mesh_quad if quad_flag else result_mesh
                    logging.info(
                        f"[step6] use_quad_remesh job_id={job_id}, quad_flag={quad_flag}, result_mesh_quad={result_mesh_quad}"
                    )


            # convert to glb
            if success_flag and result_mesh and use_output_glb and os.path.splitext(result_mesh)[-1] != ".glb":
                logging.info(f"[step2.2] convert to glb, for webui coarse vis")
                # success_flag, result_mesh = self.postprocess(job_id, result_mesh, mesh_post_processing="move-up-y", new_job_id=True)
                success_flag, result_mesh = self.postprocess(job_id, result_mesh, mesh_post_processing="move-up-y-smooth", new_job_id=True)
            logging.info(f"[DONE] main_call_step1_generate_geom_mesh once, job_id={job_id} success_flag={success_flag}, result_mesh={result_mesh}")

            # render gif
            gif_local_path = ""    # TODO
            return success_flag, result_mesh, gif_local_path

        except Exception as e:
            logging.error(
                f"[ERROR!!!!] main_call_step1_generate_geom_mesh error, job_id={job_id}, input prompt={prompt}; in_image_path_list={in_image_path_list}, error:\n{e}"
            )
            traceback.print_exc()
            self.update_task_redis(job_id, run_flag="finished", step_mode="step1")
            return False, "", ""

    def main_call_step2_mesh_gen_texture(
        self,
        job_id,
        prompt=None,
        gen_mesh_type="clay",
        texture_type="consistent_tex",
        use_mesh2image=False,
        use_d2rgb=False,
        use_imgsr=True,
        use_texrefine=False,
        use_output_glb=True,
        use_quad_remesh=True,
    ):
        """blocking mode, refine coarse mesh, run d2rgb + sr + step4_texture(baking) use ~1min

        Args:
            job_id(string), uuid
            prompt(string): 输入文字提示词.
            gen_mesh_type(string): 多图生Mesh的方法, 可以选 "crman" or "lrm"
            texture_type(string): 纹理生成方式, 可以选 consistent_tex, "baking", "syncmvd" 或 "texture_baking" 
            use_d2rgb(bool): 是否d2rgb
            use_imgsr(bool): 是否超分mvimg
            use_texrefine(bool): 是否sdxl_texrefine
            use_output_glb(bool): 是否转成glb输出
            use_quad_remesh
        Returns:
            success_flag=T/F, result_mesh=输出mesh路径, gif_local_path
        """
        try:
            if texture_type == "consistent_tex":
                # update redis flag in tex server.
                logging.info(f"[step2.1] with consistent_tex begin {job_id}")
                mesh_result_dict = self.consistent_tex_interface.blocking_call_consistent_tex(job_id)
                mesh_value_dict = mesh_result_dict[job_id][self.consistent_tex_interface.service_name]
                success_flag = mesh_value_dict["success"]
                result_mesh = mesh_value_dict["out_glb"]
                if not success_flag:
                    raise ValueError(f"[Warn] consistent_tex error with {json.dumps(mesh_result_dict)}")
            else:
                #old pipeline
                # self.update_task_redis(job_id, run_flag="running", step_mode="step2")
                # step3.0: read mesh_value_dict from redis, remove old glb in redis
                parse_flag, mesh_value_dict = self.parse_coarse_mesh_result(job_id, gen_mesh_type=gen_mesh_type)
                if not parse_flag:
                    logging.info(f"ERROR parse_coarse_mesh_result failed, job_id = {job_id}")
                    return False, None, ""
                mesh_result_dict = mesh_value_dict['result']

                ## add mesh2image if text input
                if use_mesh2image:
                    mesh_dir = mesh_result_dict['parameter']['obj_dir']
                    in_mesh_path = os.path.join(mesh_dir, 'obj_mesh_mesh.obj')

                    out_image_path = os.path.join(os.path.dirname(mesh_dir), "mesh2image.png")
                    # TODO
                    mesh2image_extra_params = {
                        "mesh_process": "keep_raw",
                    }
                    mesh2image_dict = self.mesh2image_interface.blocking_call_mesh2image(job_id,
                                                        in_mesh_path,
                                                        prompt=prompt,
                                                        in_image_path_list=None, # TODO
                                                        out_image_path=out_image_path,
                                                        mesh2image_extra_params=mesh2image_extra_params)
                    temp_dict = mesh2image_dict[job_id][self.mesh2image_interface.service_name]
                    if not temp_dict["success"]:
                        feedback = temp_dict["result"]["feedback"]
                        logging.info(f"ERROR mesh2image failed, job_id = {job_id}, feedback={feedback}")
                        return False, None, ""

                # step3.1 (optional) d2rgb
                if use_d2rgb:
                    logging.info(f"[step3.1] with use_d2rgb")
                    mesh_result_dict = self.d2rgb_interface.blocking_call_d2rgb(job_id)
                    mesh_value_dict = mesh_result_dict[job_id][self.d2rgb_interface.service_name]
                    if not mesh_value_dict["success"]:
                        raise ValueError(f"[Warn] d2rgb error with {json.dumps(mesh_result_dict)}")
                    print('run d2rgb done')

                # step3.2 (optional) mv img sr
                if use_imgsr:
                    logging.info(f"[step3.2] with use_imgsr. mv img sr")
                    in_img_path = mesh_value_dict["result"]["parameter"]["image_npy_path"]
                    up_scale = 4

                    out_img_path = os.path.join(os.path.dirname(in_img_path), "imgsr")  # 输出npy的文件夹路径 TODO:最好改下名字
                    imgsr_result_dict = self.imgsr_interface.blocking_call_mvimg_imgsr(job_id,
                                                                                    in_img_path=in_img_path,
                                                                                    out_img_path=out_img_path,
                                                                                    up_scale=up_scale)
                    if not imgsr_result_dict[job_id][self.imgsr_interface.service_name]["success"]:
                        raise ValueError(f"[ERROR] use_imgsr but imgsr error with {json.dumps(imgsr_result_dict)}")

                    imgsr_image_npy_path = imgsr_result_dict[job_id][self.imgsr_interface.service_name]["result"][0]
                    mesh_value_dict["result"]["parameter"]["image_npy_path"] = imgsr_image_npy_path

                # step4. generate texture
                success_flag, result_mesh = self.step4_texture(job_id,
                                                            mesh_value_dict,
                                                            high_res=False,
                                                            prompt=prompt,
                                                            texture_type=texture_type,
                                                            use_texrefine=use_texrefine)

            # step5. convert to glb
            if success_flag and result_mesh and use_output_glb and os.path.splitext(result_mesh)[-1] != ".glb":
                logging.info(f"[step5] convert to glb with move-up-y-smooth")
                success_flag, result_mesh = self.postprocess(job_id, result_mesh, mesh_post_processing="move-up-y-smooth")

            # 6. quad_remesh TODO
            if texture_type != "consistent_tex" and use_quad_remesh and success_flag:
                logging.info(f"begin quad_remesh, job_id = {job_id}")
                run_in_cos, need_run = True, True
                if run_in_cos:
                    quad_in_mesh = os.path.join("/mnt/aigc_bucket_4/pandorax/quad_remesh", job_id,
                                                os.path.basename(result_mesh))
                    job_dir = os.path.dirname(quad_in_mesh)
                    success = self.cos_client.upload_abs_cos_once(result_mesh, quad_in_mesh)
                    if not success:
                        logging.error(f"[Warn] quad run_in_cos but upload_abs_cos_once failed job_id={job_id} ")
                        need_run = False
                else:
                    quad_in_mesh = result_mesh
                    job_dir = os.path.dirname(os.path.dirname(result_mesh))
                if need_run:
                    output_file = os.path.join(job_dir, "quad_remesh/quad_mesh.glb")
                    final_result_dict = self.quad_remesh_interface.blocking_call_quad_remesh(
                        job_id,
                        job_dir,
                        quad_in_mesh,
                        output_file,
                    )
                    quad_flag, result_mesh_quad = self.parse_final_result_dict(
                        job_id, final_result_dict, self.quad_remesh_interface.service_name)
                    result_mesh = result_mesh_quad if quad_flag else result_mesh
                    logging.info(
                        f"[step6] use_quad_remesh job_id={job_id}, quad_flag={quad_flag}, result_mesh_quad={result_mesh_quad}"
                    )

            gif_local_path = ""
            if success_flag:
                # render gif
                try:
                    gif_local_path = result_mesh.replace(".glb", ".gif")
                    suc_gif, gif_local_path = self.gputool_interface.blocking_call_render_gif(
                        job_id, result_mesh, gif_local_path)
                    if not suc_gif:
                        logging.error(f"[ERROR] render_gif failed. job_id={job_id}, gif_local_path={gif_local_path}")
                        gif_local_path = ""
                except Exception as e:
                    logging.error(f"render gif failed, job_id={job_id}")
                    traceback.print_exc()
                    
            self.update_task_redis(job_id, run_flag="finished", step_mode="step2")

            logging.info(f"[DONE] main_call_step2_mesh_gen_texture once, job_id={job_id}, success_flag={success_flag}, result_mesh={result_mesh}")
            return success_flag, result_mesh, gif_local_path

        except Exception as e:
            logging.error(f"[ERROR!!!!] main_call_step2_mesh_gen_texture error, job_id={job_id}, input: {prompt}, error:\n{e}")
            traceback.print_exc()
            self.update_task_redis(job_id, run_flag="finished", step_mode="step2")
            return False, None, ""

    def preprocess_mesh(self, job_id, in_mesh_path, out_dir=None):
        suc_flag, new_in_mesh_path = False, None
        try:
            in_pre, in_ext = os.path.splitext(in_mesh_path)
            if out_dir is None:
                new_in_mesh_path = in_pre + ".obj"
            else:
                # TODO
                new_in_mesh_path = os.path.join(out_dir, "geom/raw_mesh.obj")
            job_id_use = f"{job_id}_preprocess_mesh"
            result_dict = self.blender_cvt_interface.blocking_call_anything_to_obj(job_id_use, in_mesh_path, new_in_mesh_path)
            cvt_dict = result_dict[job_id_use][self.blender_cvt_interface.service_name]
            if not cvt_dict["success"]:
                err_str = f"[ERROR] preprocess_mesh failed,  job_id:\n{job_id}, job_id_use:\n{job_id_use}"
                logging.error(err_str)
                return False, err_str
            return True, cvt_dict["result"]

        except Exception as e:
            logging.error(f"[ERROR!!!!] preprocess_mesh error,  error:\n{e}")
            traceback.print_exc()

        return suc_flag, new_in_mesh_path

    def call_generate_texture_pipe(
        self,
        job_id,
        in_mesh_path,
        prompt=None,
        in_image_path_list=None,
        out_dir=None,
        texture_type="texall",
        use_imgsr=True,
        use_texrefine=False,
        use_output_glb=True,
        use_quad_remesh=False,
        extra_params=dict(),
    ):
        """blocking mode, generate texture with text, run mesh2image + d2rgb + sr + step4_texture

        Args:
            job_id(string), uuid
            in_mesh_path
            prompt(string): 输入文字提示词.
            in_image_path_list([str]): 输入图片 or None
            out_dir
            texture_type(string): 纹理生成方式, 可以选 "baking", "syncmvd" 或 "texture_baking" 
            use_texrefine(bool): 是否sdxl_texrefine
            use_output_glb(bool): 是否转成glb输出
        Returns:
            success_flag=T/F, result_mesh=输出mesh路径 (如果异常，返回异常原因str), gif_local_path
        """
        try:
            # TODO
            logging.info(f"[tex-step1] with preprocess_mesh begin {job_id}")
            suc_flag, in_mesh_path = self.preprocess_mesh(job_id, in_mesh_path, out_dir=out_dir)
            if not suc_flag:
                logging.error(f"[ERROR] preprocess_mesh error, job_id={job_id}")
                return False, "preprocess_mesh error", ""

            logging.info(f"[tex-step2] with mesh2image begin {job_id}")
            out_image_path = os.path.join(out_dir, "mesh2image.png")
            mesh2image_extra_params = extra_params.get("mesh2image_extra_params", dict())
            mesh2image_dict = self.mesh2image_interface.blocking_call_mesh2image(job_id,
                                                in_mesh_path,
                                                prompt=prompt,
                                                in_image_path_list=in_image_path_list,
                                                out_image_path=out_image_path,
                                                mesh2image_extra_params=mesh2image_extra_params)
            temp_dict = mesh2image_dict[job_id][self.mesh2image_interface.service_name]
            if not temp_dict["success"]:
                feedback = temp_dict["result"]["feedback"]
                logging.info(f"ERROR mesh2image failed, job_id = {job_id}, feedback={feedback}")
                return False, None, ""
            print('run mesh2image done')

            in_condition_path = temp_dict["result"]["out_path"]
            new_mesh_path = temp_dict["result"].get("out_obj", in_mesh_path)
            job_id_dir = os.path.dirname(in_condition_path)

            if texture_type == "consistent_tex":
                logging.info(f"[tex-step3.1] with consistent_tex begin {job_id}")
                out_dir = os.path.join(job_id_dir, 'texture_mesh')
                mesh_result_dict = self.consistent_tex_interface.blocking_call_consistent_tex_direct(
                    job_id, new_mesh_path, in_condition_path, out_dir)
                mesh_value_dict = mesh_result_dict[job_id][self.consistent_tex_interface.service_name]
                success_flag = mesh_value_dict["success"]
                result_mesh = mesh_value_dict["out_glb"]
                if not success_flag:
                    raise ValueError(f"[Warn] consistent_tex error with {json.dumps(mesh_result_dict)}")
            else:
                # step3.1 (optional) d2rgb
                logging.info(f"[step3.1] with use_d2rgb")
                mesh_result_dict = self.d2rgb_interface.blocking_call_d2rgb_direct(
                    job_id,
                    new_mesh_path,
                    in_condition_path,
                    "model_texturing",
                    os.path.join(job_id_dir, "d2rgb/out"),
                    os.path.join(job_id_dir, "d2rgb/vis"),
                )
                mesh_value_dict = mesh_result_dict[job_id][self.d2rgb_interface.service_name]
                if not mesh_value_dict["success"]:
                    raise ValueError(f"[Warn] d2rgb error with {json.dumps(mesh_result_dict)}")
                print('run d2rgb done')

                # step3.2 (optional) mv img sr
                if use_imgsr:
                    logging.info(f"[step3.2] with use_imgsr. mv img sr")
                    in_img_path = mesh_value_dict["result"]["parameter"]["image_npy_path"]

                    out_img_path = os.path.join(os.path.dirname(in_img_path), "imgsr")  # 输出npy的文件夹路径 TODO:最好改下名字
                    imgsr_result_dict = self.imgsr_interface.blocking_call_mvimg_imgsr(job_id,
                                                                                    in_img_path=in_img_path,
                                                                                    out_img_path=out_img_path,
                                                                                    )
                    if not imgsr_result_dict[job_id][self.imgsr_interface.service_name]["success"]:
                        raise ValueError(f"[ERROR] use_imgsr but imgsr error with {json.dumps(imgsr_result_dict)}")

                    imgsr_image_npy_path = imgsr_result_dict[job_id][self.imgsr_interface.service_name]["result"][0]
                    # assert os.path.exists(
                    #     imgsr_image_npy_path
                    # ), f"[ERROR] imgsr success but can not find output imgsr_image_npy_path: {imgsr_image_npy_path}"
                    mesh_value_dict["result"]["parameter"]["image_npy_path"] = imgsr_image_npy_path

                # step4. generate texture
                success_flag, result_mesh = self.step4_texture(job_id,
                                                            mesh_value_dict,
                                                            high_res=False,
                                                            prompt=prompt,
                                                            texture_type=texture_type,
                                                            use_texrefine=use_texrefine,)

            # step5. convert to glb
            if success_flag and result_mesh and use_output_glb and os.path.splitext(result_mesh)[-1] != ".glb":
                logging.info(f"[step5] convert to glb")
                success_flag, result_mesh = self.postprocess(job_id, result_mesh, mesh_post_processing="move-up-y-smooth",
                                                             new_job_id=True)

            # render gif
            gif_local_path = ""
            try:
                gif_local_path = result_mesh.replace(".glb", ".gif")
                suc_gif, gif_local_path = self.gputool_interface.blocking_call_render_gif(
                    job_id, result_mesh, gif_local_path)
                if not suc_gif:
                    gif_local_path = ""
                    logging.info(f"[WARN] render gif failed job_id={job_id}")
            except Exception as e:
                logging.info(f"[fatal Error] render gif failed job_id={job_id} {e}")
                traceback.print_exc()
                
            self.update_task_redis(job_id, run_flag="finished", step_mode="step2")

            logging.info(f"[DONE] call_generate_texture_pipe once, success_flag={success_flag}, result_mesh={result_mesh}")
            return success_flag, result_mesh, gif_local_path

        except Exception as e:
            logging.error(f"[ERROR!!!!] call_generate_texture_pipe error, job_id={job_id}, input: {prompt}, error:\n{e}")
            traceback.print_exc()
            return False, str(e), ""

    def get_pipe_params(self, reload_pipe_json=None):
        try:
            if reload_pipe_json is not None:
                self.pipe_json, self.pipe_dict = self.load_cfg(reload_pipe_json)
                logging.info(f'update pipe dict from {reload_pipe_json}')

        except Exception as e:
            print(f"[ERROR] get_pipe_params failed. reload_pipe_json={reload_pipe_json}, use default")

        gen_mvimg_type = self.pipe_dict.get("gen_mvimg_type", "t2i_z123")
        gen_mesh_type = self.pipe_dict.get("gen_mesh_type", "mv2mesh")
        texture_type = self.pipe_dict.get("texture_type", "baking")
        use_d2rgb = self.pipe_dict.get("use_d2rgb", False)
        use_imgsr = self.pipe_dict.get("use_imgsr", True)
        use_texrefine = self.pipe_dict.get("use_texrefine", False)
        extra_params = self.pipe_dict.get("extra_params", dict())
        return gen_mvimg_type, gen_mesh_type, texture_type, use_d2rgb, use_imgsr, use_texrefine, extra_params

    def step1_mv(
        self,
        job_id,
        prompt=None,
        in_image_path_list=None,
        out_save_dir=None,
        gen_mvimg_type="t2i_mmd",
        extra_params={},
    ):
        """step1, input text or image, generate multi-view images (npy)

        Args:
            job_id(string), uuid
            prompt(string): 输入文字提示词.
            in_image_path_list(list of string): list of 输入图片绝对路径. 
            out_save_dir(string): 输出文件夹的根目录. 可以指定路径或者None
            gen_mvimg_type(string): 如果用输入文字控制,= t2i_mmd, t2i_z123 or t2mvd, 如果用输入图像控制可以选 "z123" 或 "w3d" 

        Returns:
            mv_parameter_dict, as input of step2_mesh. Example:
            {
                "job_id": "b2570b32-d7db-4a86-bec5-9e878af8865e",
                "image_path": "/aigc_cfs_gdp/neoshang/data/validation/mario.png",
                "wh_ratio": 0.8,
                "cfg_scale": 3.0,
                "step_num": 50,
                "image_npy_path": "/aigc_cfs_gdp/sz/result/pipe_test/b2570b32-d7db-4a86-bec5-9e878af8865e/mario_color.npy",
                "normal_npy_path": "/aigc_cfs_gdp/sz/result/pipe_test/b2570b32-d7db-4a86-bec5-9e878af8865e/mario_normal.npy",
                "ref_img_path": "/aigc_cfs_gdp/sz/result/pipe_test/b2570b32-d7db-4a86-bec5-9e878af8865e/mario_ref.png",
                "job_id_dir": "/aigc_cfs_gdp/sz/result/pipe_test/b2570b32-d7db-4a86-bec5-9e878af8865e"
            }               
        """
        try:
            # step1. generate multi-view images
            logging.info(f"[step1] with {gen_mvimg_type} begin. generate multi-view images\n job_id={job_id}")
            if out_save_dir and not os.path.exists(out_save_dir):
                os.makedirs(out_save_dir, exist_ok=True)
            if gen_mvimg_type == "z123":
                result_dict = self.zero123plus_interface.blocking_call_img_to_mvimg(job_id,
                                                                                    in_image_path_list,
                                                                                    out_save_dir=out_save_dir)
                print('result_dict zero123plus:\n', result_dict)
                mv_parameter_dict = result_dict[job_id][gen_mvimg_type]['parameter']
            elif gen_mvimg_type == "imagedream":
                result_dict = self.imagedream_interface.blocking_call_img_to_mvimg(
                    job_id,
                    in_image_path_list,  # 
                    out_save_dir=out_save_dir)
                print('result_dict imagedream:\n', result_dict)
                mv_parameter_dict = result_dict[job_id][gen_mvimg_type]['parameter']
            elif gen_mvimg_type == "w3d":
                result_dict = self.mmd_interface.blocking_call_img_to_mvimg(job_id,
                                                                            in_image_path_list[0],  # TODO mmd needsupport list
                                                                            out_save_dir=out_save_dir)
                print('result_dict mmd:\n', result_dict)
                mv_parameter_dict = result_dict[job_id][gen_mvimg_type]['result']
            # text input version
            elif gen_mvimg_type == "t2mvd":
                result_dict = self.t2mvd_interface.blocking_call_text_to_mvimg(job_id,
                                                                               prompt,
                                                                               out_save_dir=out_save_dir)
                print('result_dict t2mvd:\n', result_dict)
                mv_parameter_dict = result_dict[job_id][gen_mvimg_type]['parameter']
            elif gen_mvimg_type == "t2i_z123":
                if not prompt and (in_image_path_list is not None) and (len(in_image_path_list) >= 1):
                    in_image_path_list_ = in_image_path_list
                    prompt_ = ""
                    logging.info(f"use image input mode with gen_mvimg_type={gen_mvimg_type}")
                else:
                    image_path = self.call_text_to_image(job_id, prompt, out_save_dir, extra_params=extra_params)
                    in_image_path_list_ = [image_path]
                    prompt_ = prompt

                # TODO, HACK make sure update running
                self.update_task_redis(job_id, run_flag="running", step_mode="step1")

                result_dict = self.zero123plus_interface.blocking_call_img_to_mvimg(job_id,
                                                                                    in_image_path_list_,
                                                                                    prompt=prompt_,
                                                                                    out_save_dir=out_save_dir)
                print('result_dict t2i_z123--z123:\n', result_dict)
                mv_parameter_dict = result_dict[job_id]["z123"]['parameter']
                result_dict[job_id][gen_mvimg_type] = {'success' : result_dict[job_id]["z123"]["success"]}

            elif gen_mvimg_type == "t2i_imagedream":
                if not prompt and (in_image_path_list is not None) and (len(in_image_path_list) >= 1):
                    in_image_path_list_ = in_image_path_list
                    prompt_ = ""
                    logging.info(f"use image input mode with gen_mvimg_type={gen_mvimg_type}")
                else:
                    image_path = self.call_text_to_image(job_id, prompt, out_save_dir, extra_params=extra_params)
                    in_image_path_list_ = [image_path]
                    prompt_ = prompt

                # TODO, HACK make sure update running
                self.update_task_redis(job_id, run_flag="running", step_mode="step1")

                result_dict = self.imagedream_interface.blocking_call_img_to_mvimg(job_id,
                                                                                    in_image_path_list_,
                                                                                    prompt=prompt_,
                                                                                    out_save_dir=out_save_dir)
                print('result_dict t2i_imagedream--imagedream:\n', result_dict)
                mv_parameter_dict = result_dict[job_id]["imagedream"]['parameter']
                result_dict[job_id][gen_mvimg_type] = {'success' : result_dict[job_id]["imagedream"]["success"]}

            elif gen_mvimg_type == "t2i_mmd":
                if not prompt and (in_image_path_list is not None) and (len(in_image_path_list) >= 1):
                    image_path = in_image_path_list[0]
                    logging.info(f"use image input mode with gen_mvimg_type={gen_mvimg_type}")
                else:
                    image_path = self.call_text_to_image(job_id, prompt, out_save_dir, extra_params=extra_params)

                # TODO, HACK make sure update running
                self.update_task_redis(job_id, run_flag="running", step_mode="step1")

                result_dict = self.mmd_interface.blocking_call_img_to_mvimg(job_id,
                                                                                    image_path,
                                                                                    out_save_dir=out_save_dir)
                print('result_dict t2i_mmd--mmd:\n', result_dict)
                mv_parameter_dict = result_dict[job_id]["w3d"]['parameter']
                result_dict[job_id][gen_mvimg_type] = {'success' : result_dict[job_id]["w3d"]["success"]}
            elif gen_mvimg_type == "skip":
                # TODO. temp. need refine
                if not prompt and (in_image_path_list is not None) and (len(in_image_path_list) >= 1):
                    in_image_path_list_ = in_image_path_list
                    prompt_ = ""
                    logging.info(f"use image input mode with gen_mvimg_type={gen_mvimg_type}")
                else:
                    image_path = self.call_text_to_image(job_id, prompt, out_save_dir, extra_params=extra_params)
                    in_image_path_list_ = [image_path]
                    prompt_ = prompt
                    logging.info(f"use text input mode and t2i with gen_mvimg_type={gen_mvimg_type}")

                success_flag = True
                job_id_dir = os.path.join(out_save_dir, job_id)

                mv_parameter_dict = {
                    "job_id": job_id,
                    "in_image_path_list": in_image_path_list_,
                    "image_path_list": in_image_path_list_,
                    "image_mask_path_list": in_image_path_list_,
                    "image_npy_path": in_image_path_list_,
                    "image_npy_path_rescale": in_image_path_list_,
                    "job_id_dir": job_id_dir,
                }
                result_dict = {job_id : {gen_mvimg_type: {
                    'success' : success_flag,
                    "parameter": mv_parameter_dict
                }}}

            # elif gen_mvimg_type == "t2i_skip":
            #     # TODO.
            #     success_flag, image_npy_path = True, in_image_path_list
            #     job_id_dir = os.path.join(out_save_dir, job_id)

            #     mv_parameter_dict = {
            #         "job_id": job_id,
            #         "in_image_path_list": in_image_path_list,
            #         "image_path_list": in_image_path_list,
            #         "image_mask_path_list": in_image_path_list,
            #         "image_npy_path": image_npy_path,
            #         "image_npy_path_rescale": image_npy_path,
            #         "job_id_dir": job_id_dir,
            #     }
            #     result_dict = {job_id : {gen_mvimg_type: {
            #         'success' : success_flag,
            #         "parameter": mv_parameter_dict
            #     }}}
            else:
                raise ValueError(f"invalid gen_mvimg_type:{gen_mvimg_type}")


            if not result_dict[job_id][gen_mvimg_type]['success']:
                raise ValueError(f"[Warn] gen_mvimg error with {json.dumps(result_dict)}")

            logging.info(f"[step1] with {gen_mvimg_type} done")
            return mv_parameter_dict

        except Exception as e:
            logging.error(
                f"[ERROR_step1] step1_mv error, input job_id={job_id}. prompt={prompt}: in_image_path_list={in_image_path_list}, error:\n{e}"
            )
            return None

    def step2_mesh(self, job_id, mv_parameter_dict, gen_mesh_type="lrm"):
        """_summary_

        Args:
            job_id(string), uuid
            mv_parameter_dict(dict),  out from step1.
            gen_mesh_type(string): 多图生Mesh的方法, 可以选 "crman" or "lrm" or "mv2mesh"

        Returns:
            mesh_value_dict: input of step3/4 sr/baking. Example:
        {
            "service_name": "lrm",
            "success": true,
            "result": {
                "service_name": "lrm",
                "parameter": {
                    "job_id": "b2570b32-d7db-4a86-bec5-9e878af8865e",
                    "image_path": "/aigc_cfs_gdp/neoshang/data/validation/mario.png",
                    "text_prompt": null,
                    "image_npy_path": "/aigc_cfs_gdp/sz/result/pipe_test/b2570b32-d7db-4a86-bec5-9e878af8865e/mario_color.npy",
                    "normal_npy_path": "/aigc_cfs_gdp/sz/result/pipe_test/b2570b32-d7db-4a86-bec5-9e878af8865e/mario_normal.npy",
                    "job_id_dir": "/aigc_cfs_gdp/sz/result/pipe_test/b2570b32-d7db-4a86-bec5-9e878af8865e",
                    "obj_dir": "/aigc_cfs_gdp/sz/result/pipe_test/b2570b32-d7db-4a86-bec5-9e878af8865e/obj_dir",
                    "pre_service": "w3d"
                }
            }
        }
        """
        mesh_value_dict = None
        try:
            # step2. generate mesh
            logging.info(f"[step2] with {gen_mesh_type} begin. generate mesh")
            if gen_mesh_type == "lrm":
                mesh_result_dict = self.lrm_interface.blocking_call_mvimg_to_mesh(job_id, mv_parameter_dict)
                mesh_value_dict = mesh_result_dict[job_id][self.lrm_interface.service_name]

            elif gen_mesh_type == "mv2mesh":
                mesh_result_dict = self.mv2mesh_interface.blocking_call_mvimg_to_mesh(job_id, mv_parameter_dict)
                mesh_value_dict = mesh_result_dict[job_id][self.mv2mesh_interface.service_name]

            elif gen_mesh_type == "clay":
                mesh_result_dict = self.clay_interface.blocking_call_mvimg_to_mesh(job_id, mv_parameter_dict)
                mesh_value_dict = mesh_result_dict[job_id][self.clay_interface.service_name]

            elif gen_mesh_type == "clay_1img":
                mesh_result_dict = self.clay_1img_interface.blocking_call_mvimg_to_mesh(job_id, mv_parameter_dict)
                mesh_value_dict = mesh_result_dict[job_id][self.clay_1img_interface.service_name]

            elif gen_mesh_type == "crman":
                mesh_result_dict = self.crman_interface.blocking_call_mvimg_to_mesh(job_id, mv_parameter_dict)
                mesh_value_dict = mesh_result_dict[job_id][self.crman_interface.service_name]

            else:
                raise ValueError(f"invalid gen_mesh_type:{gen_mesh_type}")
            if not mesh_value_dict["success"]:
                raise ValueError(f"[Warn] gen_mesh {gen_mesh_type} failed with {json.dumps(mesh_result_dict)}")

            logging.info(f"[step2] with {gen_mesh_type} done")
            return mesh_value_dict
        except Exception as e:
            logging.error(
                f"[ERROR_step2] step2_mesh error, input job_id={job_id}. gen_mesh_type={gen_mesh_type},  error:\n{e}"
            )
            traceback.print_exc()
            print('mv_parameter_dict ', mv_parameter_dict)
            print('mesh_value_dict ', mesh_value_dict)
            # breakpoint()
            return None

    def step4_texture(self, job_id, mesh_value_dict, high_res=False, prompt=None, texture_type="baking",
                      use_texrefine=False):
        """_summary_

        Args:
            job_id(string), uuid
            mesh_value_dict(dict), out from mesh, or (option d2rgb/sr)
            high_res for baking
            prompt(string), if use syncmvd
            texture_type(string): 纹理生成方式, 可以选 "texall", "baking" "syncmvd" 或 "texture_baking" 
        Returns:
            success_flag=T or F, 
            result_mesh = out mesh path
        """
        try:
            logging.info(f"[step4] with {texture_type}. generate texture")
            if texture_type == "baking":
                # fast baking + verts2tex
                baking_result_dict = self.fast_baking_interface.blocking_call_texture_baking(job_id,
                                                                                             mesh_value_dict,
                                                                                             high_res=high_res)
                value_dict = baking_result_dict[job_id][self.fast_baking_interface.service_name]
                if not value_dict["success"]:
                    logging.error(f"[ERROR_step4] run fast baking failed with job_id={job_id}")
                    return False, None

                baking_dir = value_dict["result"]["out_obj_dir"]
                source_mesh_path = os.path.join(baking_dir, "mesh.obj")
                baking_name = os.path.basename(baking_dir)
                output_mesh_folder = os.path.join(os.path.dirname(baking_dir), "verts2tex", baking_name)
                final_result_dict = self.verts2tex_interface.blocking_call_verts2tex(
                    job_id, source_mesh_path, output_mesh_folder, "tex_mesh")

            elif texture_type == "texall":
                # need d2rgb firstly TODO
                parameter_dict = mesh_value_dict["result"]["parameter"]
                final_result_dict = self.texall_interface.blocking_call_interface_texall(
                    job_id,
                    parameter_dict["in_obj_path"],
                    parameter_dict["image_npy_path"],
                    out_dir=os.path.join(os.path.dirname(os.path.dirname(parameter_dict["out_dir"])), "texall"),
                    use_texrefine=use_texrefine,
                )
            elif texture_type == "texall2":
                # need d2rgb firstly TODO
                parameter_dict = mesh_value_dict["result"]["parameter"]
                out_dir = os.path.join(os.path.dirname(os.path.dirname(parameter_dict["out_dir"])), "texall")
                final_result_dict = self.texall2_interface.blocking_call_interface(
                    job_id,
                    parameter_dict["in_obj_path"],
                    parameter_dict["image_npy_path"],
                    out_dir,
                    os.path.join(os.path.dirname(out_dir), "d2rgb/vis/out.png"),
                    f"out/{job_id}.obj",
                    "human_6views_1",
                )

            elif texture_type == "texbakeinpaint":
                # need d2rgb firstly TODO
                parameter_dict = mesh_value_dict["result"]["parameter"]
                final_result_dict = self.texbakeinpaint_interface.blocking_call_texbakeinpaint(
                    job_id,
                    parameter_dict["in_obj_path"],
                    parameter_dict["image_npy_path"],
                    out_dir=os.path.join(os.path.dirname(os.path.dirname(parameter_dict["out_dir"])), "texbakeinpaint"),
                )

            # old pipeline
            elif texture_type == "texture_baking":
                final_result_dict = self.texture_baking_interface.blocking_call_texture_baking(job_id, mesh_value_dict)
            elif texture_type == "syncmvd":
                lrm_mesh_path = os.path.join(mesh_value_dict["result"]["parameter"]["obj_dir"], "obj_mesh_mesh.obj")
                out_objs_dir = os.path.join(mesh_value_dict["result"]["parameter"]['job_id_dir'], "syncmvd")
                final_result_dict = self.texture_syncmvd_interface.blocking_call_text_texture(job_id,
                                                                                              lrm_mesh_path,
                                                                                              prompt,
                                                                                              out_objs_dir=out_objs_dir)
            logging.info(f"[step4] with {texture_type} done")

            success_flag, result_mesh = self.parse_final_result_dict(job_id, final_result_dict, texture_type)


            # step4.2 (optional) sdxl tex_refine, old pipeline
            if success_flag and texture_type == "texbakeinpaint" and use_texrefine:
                logging.info(f"[step4.2] sdxl tex_refine")
                texrefine_result_dict = self.sdxl_texrefine_interface.blocking_call_sdxl_inpaint(
                    job_id, os.path.dirname(result_mesh))
                texrefine_data = texrefine_result_dict[job_id][self.sdxl_texrefine_interface.service_name]
                success_flag, result_mesh = texrefine_data["success"], os.path.join(texrefine_data["output_dir"],
                                                                                    "textured.obj")

            return success_flag, result_mesh

        except Exception as e:
            logging.error(
                f"[ERROR_step4] step4_texture error, input job_id={job_id}. texture_type={texture_type}: mesh_value_dict=\n {json.dumps(mesh_value_dict)} error:\n{e}"
            )
            traceback.print_exc()
            return False, None

    def parse_final_result_dict(self, job_id, final_result_dict, final_service_name):
        """Unify the final results of different texture methods into a consistent output: success_flag and result_mesh.

        Args:
            job_id: _description_
            final_result_dict: _description_
            final_service_name: baking, quad_remesh texture_baking or syncmvd

        Returns:
            success_flag=T or F, 
            result_mesh = out mesh path
        """
        try:
            if final_service_name == "baking":
                value_dict = final_result_dict[job_id][self.verts2tex_interface.service_name]
                success_flag = value_dict["success"]
                result_mesh = value_dict["result"]
            elif final_service_name == "texall":
                value_dict = final_result_dict[job_id][self.texall_interface.service_name]
                success_flag = value_dict["success"]
                result_mesh = value_dict["out_glb"]
            elif final_service_name == "texall2":
                value_dict = final_result_dict[job_id][self.texall2_interface.service_name]
                success_flag = value_dict["success"]
                result_mesh = value_dict["out_glb"]
            elif final_service_name == "texbakeinpaint":
                value_dict = final_result_dict[job_id][self.texbakeinpaint_interface.service_name]
                success_flag = value_dict["success"]
                result_mesh = value_dict["result"]
            elif final_service_name == "quad_remesh":
                value_dict = final_result_dict[job_id][self.quad_remesh_interface.service_name]
                success_flag = value_dict["success"]
                result_mesh = value_dict["output_file"]
            elif final_service_name == "texture_baking":
                value_dict = final_result_dict[job_id][self.texture_baking_interface.service_name]
                success_flag = value_dict["success"]
                result_mesh = os.path.join(value_dict["out_obj_dir"], "mesh.obj")
            elif final_service_name == "syncmvd":
                value_dict = final_result_dict[job_id][self.texture_syncmvd_interface.service_name]
                success_flag = value_dict["success"]
                result_mesh = value_dict["result"][0]  # sync will return list, len=1

            if not success_flag or not result_mesh:
                return False, []
            return success_flag, result_mesh
        except Exception as e:
            logging.error(
                f"[ERROR!!!!] parse_final_result_dict error, job_id: {job_id}, final_result_dict:{json.dumps(final_result_dict)}\n error:\n{e}"
            )
            return False, []

    def postprocess(self, job_id, result_mesh, mesh_post_processing="move-up-y", new_job_id=False, out_result_glb=None):
        """convert obj to glb

        Args:
            job_id: _description_
            result_mesh: raw obj

        Returns:
            success_flag, out_result_mesh=out glb
        """
        job_id_ = job_id + "_" + init_job_id() if new_job_id else job_id
        try:
            assert os.path.exists(result_mesh), f"can not find result_mesh={result_mesh} when postprocess"
            if out_result_glb is None:
                out_result_glb = os.path.join(os.path.dirname(result_mesh), f"out/{job_id}.glb")
            result_dict = self.blender_cvt_interface.blocking_call_obj_to_glb(job_id_,
                                                                              result_mesh,
                                                                              out_result_glb,
                                                                              mesh_post_processing=mesh_post_processing)
            value_dict = result_dict[job_id_][self.blender_cvt_interface.service_name]
            success_flag = value_dict["success"]
            out_result_mesh = value_dict["result"]

            logging.info(
                f"postprocess done, success_flag={success_flag}, input result_mesh={result_mesh}, out out_result_mesh={out_result_mesh}"
            )

            if not success_flag or not out_result_mesh:
                return False, ""
            return success_flag, out_result_mesh
        except Exception as e:
            logging.error(f"[ERROR!!!!] postprocess error, job_id_: {job_id_}, result_mesh:{result_mesh}\n error:\n{e}")
            return False, ""

    def call_text_to_image(self, job_id, prompt, out_save_dir, extra_params={}):
        t2i_extra_params = extra_params.get("t2i_extra_params", {})
        result_dict = self.text2img_interface.blocking_call_text_to_img(job_id,
                                                                        prompt,
                                                                        out_save_dir=out_save_dir,
                                                                        t2i_extra_params=t2i_extra_params)
        print('result_dict text2img:\n', result_dict)
        if not result_dict[job_id][self.text2img_interface.service_name]["success"]:
            raise ValueError(f"text2img faild, job_id={job_id}")
        image_path = result_dict[job_id][self.text2img_interface.service_name]['parameter']["image_path"]
        return image_path

    def close(self):
        self.text2img_interface.close()
        self.t2mvd_interface.close()
        self.t2i_mmd_interface.close()
        self.zero123plus_interface.close()
        self.imagedream_interface.close()
        self.mmd_interface.close()

        self.lrm_interface.close()
        self.crman_interface.close()
        self.mv2mesh_interface.close()
        self.clay_interface.close()
        self.clay_1img_interface.close()

        self.mesh2image_interface.close()
        self.d2rgb_interface.close()
        self.consistent_tex_interface.close()
        self.imgsr_interface.close()

        self.texall_interface.close()
        self.texall2_interface.close()
        self.texbakeinpaint_interface.close()
        self.sdxl_texrefine_interface.close()
        self.fast_baking_interface.close()
        self.verts2tex_interface.close()
        self.texture_baking_interface.close()
        self.texture_syncmvd_interface.close()

        self.quad_remesh_interface.close()
        self.blender_cvt_interface.close()
        self.gputool_interface.close()

        if self.open_task_queue:
            self.task_queue_step1.close()
            self.task_queue_step2.close()

