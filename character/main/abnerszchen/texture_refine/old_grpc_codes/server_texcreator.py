import logging
import argparse
import os
import sys
import grpc
import texcreator_pb2
import texcreator_pb2_grpc
from concurrent import futures
import threading

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from dataset.utils_dataset import load_json
from pipe_texcreator import ObjTexCreatorPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class TexcreatorServer(texcreator_pb2_grpc.TexcreatorServicer):
    def __init__(self, grpc_cfg_json='tex_creator.json', model_key='tex_creator_human_design9'):
        """_summary_

        Args:
            grpc_cfg_json: _description_. Defaults to 'tex_creator.json'.
            model_key: TODO need select in webui
        """
        super(TexcreatorServer, self).__init__()
        self.jobs = {}
        self.job_cnt = 0
        self.lock = threading.Lock()
        self.retry = 3

        self.configs_dir = os.path.join(root_dir, 'configs')
        _grpc_cfg_json = os.path.join(self.configs_dir, grpc_cfg_json)
        assert os.path.exists(_grpc_cfg_json), f"can not find grpc_cfg_json {_grpc_cfg_json}"

        self.grpc_cfg = load_json(_grpc_cfg_json)[model_key]
        
        in_model_path = self.grpc_cfg['in_model_path']
        optim_cfg_json = os.path.join(self.configs_dir, self.grpc_cfg.get('optim_cfg', 'optim_cfg_high.json'))
        pose_json = os.path.join(root_dir, self.grpc_cfg['pose_json'])
        lrm_mode = self.grpc_cfg['lrm_mode']

        logging.info(f'begin loading {model_key} model from {in_model_path} with lrm_mode {lrm_mode}, Need a little time in T10')
        self.pipe = ObjTexCreatorPipeline(in_model_path, optim_cfg_json, pose_json, lrm_mode, device='cuda')
        logging.info(f'init tex server with grpc_cfg_json {grpc_cfg_json}')

    # common utils
    def get_job_id(self):
        """Get unique job ID after increasing the self.job_cnt

        Returns:
            int: newest job_id
        """
        self.lock.acquire()

        try:
            self.job_cnt += 1
        finally:
            self.lock.release()

        return self.job_cnt

    def finish_job(self, job_id, results):
        """Save the status and results in self.jobs

        Args:
            job_id (int): the unique ID of the job
            results (list[str]): the generated model paths
        """
        self.lock.acquire()

        self.jobs[job_id]['results'] = results
        self.jobs[job_id]['status'] = texcreator_pb2.FINISHED

        self.lock.release()

    def start_new_job(self, function, *args):
        """Start a new job and record everything in self.jobs

        Args:
            function (_type_): the function to run
            *args: the parameters pass to function

        Returns:
            int: job_id
        """
        job_id = self.get_job_id()

        self.lock.acquire()
        self.jobs[job_id] = {
            'status': texcreator_pb2.IN_PROGRESS,
        }
        self.lock.release()

        # print('start thread')
        job = threading.Thread(target=function, args=(job_id, *args))
        print('debug: thread started')
        job.start()

        self.lock.acquire()
        self.jobs[job_id]['job'] = job
        self.lock.release()

        return job_id

    # core functions
    def set_render_pose(self, job_id, pose_json, lrm_mode, select_view=[]):
        """set render pose from pose_json and lrm_mode

        Args:
            job_id (int): the unique ID of the job
            pose_json: like data/cams/cam_parameters_srender8.json
            lrm_mode: objaverse use True, human use false. TODO need rm after fix pose
            select_view: _description_. Defaults to [].

        """
        logging.info(f'set_render_pose pose_json=\"{pose_json}\"')
        for i in range(0, self.retry):
            try:
                results = self.pipe.interface_set_render_pose(pose_json, lrm_mode, select_view=select_view)

                break
            except Exception as e:
                logging.error(f'set_render_pose failed, retry={i}')
                if i < (self.retry - 1):
                    # TODO(csz) how to re-connect
                    pass

                else:
                    raise e

        logging.info(f'set_render_pose finished, pose_json=\"{pose_json}\"')
        self.finish_job(job_id, results)

    def obj_generate_tex(self, job_id, in_obj, in_condi, out_obj, out_debug_dir=None, debug_paste_condi=False):
        """render depth from in_obj, then infer SD with in_condi, get tex rgb and optim uv texture

        Args:
            job_id (int): the unique ID of the job
            in_obj: raw obj path with uv coord
            in_condi: condi img path
            out_debug_dir: save debug result if not None. Defaults to 'None'.

        Raises:
            e: _description_
        """
        logging.info(f'obj_generate_tex in_obj=\"{in_obj}\", in_condi= {in_condi}')
        for i in range(0, self.retry):
            try:
                results = self.pipe.interface_obj_generate_tex(in_obj, in_condi, out_obj, out_debug_dir=out_debug_dir, debug_paste_condi=debug_paste_condi)

                break
            except Exception as e:
                logging.error(f'obj_generate_tex failed, retry={i}')
                if i < (self.retry - 1):
                    # TODO(csz) how to re-connect
                    pass

                else:
                    raise e

        logging.info(f'obj_generate_tex  done, in_obj=\"{in_obj}\", in_condi= {in_condi}"')
        print(f'obj_generate_tex  results: ', results)
        self.finish_job(job_id, results)
        return

    # grpc protos interface functions
    def NewJob(self, request, context):
        """Check the inputs from request and start the AIGC job accordingly

        Args:
            request (JobRequest): the request object contains all the possible parameters
            context : gRPC context

        Returns:
            JobReply: The job status
        """
        try:
            print('debug request', request)
        except Exception as e:
            print('warn print request failed, skip ', e)
            
        out_debug_dir, debug_paste_condi = None, False
        if request.out_debug_dir and len(request.out_debug_dir) > 0:
            out_debug_dir = request.out_debug_dir
        if request.debug_paste_condi and len(request.debug_paste_condi) > 0 and request.debug_paste_condi == 'true':
            debug_paste_condi = True
   
        if request.task_type == texcreator_pb2.TEX_CREATOR:
            if len(request.in_obj) > 0:
                job_id = self.start_new_job(self.obj_generate_tex, request.in_obj,
                                            request.in_condi, request.out_obj, out_debug_dir, debug_paste_condi)
            else:
                return texcreator_pb2.JobReply(job_id=-1, error='Invalid parameters')
        else:
            return texcreator_pb2.JobReply(job_id=-1, error='Invalid task_type')

        return texcreator_pb2.JobReply(job_id=job_id)


    def QueryJobs(self, request, context):
        """Query the status of jobs using list of job id

        Args:
            request (JobStatusRequest): the request that contains a list of job IDs

        Returns:
            JobStatusReply: the reply that contains a list of job JobReply
        """
        rets = []
        for job_id in request.job_ids:
            if job_id in self.jobs:
                info = self.jobs[job_id]
                # print('debug query job info', info)
                ret = texcreator_pb2.JobReply(
                    job_id=job_id,
                    status=info['status'],
                    estimated_time=10,
                    results=info.get('results', ''),
                    error='')
                rets.append(ret)
                
        return texcreator_pb2.JobStatusReply(job_status=rets)

        
    def Add(self, request, context):
        """simple function for debug
        """
        return texcreator_pb2.AdditionResponse(result=request.x + request.y)


def main_server(lj_port='8080', grpc_cfg_json='tex_creator.json', model_key='tex_creator_human_design9'):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    texcreator_pb2_grpc.add_TexcreatorServicer_to_server(TexcreatorServer(grpc_cfg_json, model_key), server)
    server.add_insecure_port(f"[::]:{lj_port}")
    server.start()
    logging.info(f"Server started, listening on {lj_port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('--lj_port', type=str, default='8080')
    parser.add_argument('--grpc_cfg_json', type=str, default='tex_creator.json')
    parser.add_argument('--model_key', type=str, default='tex_creator_human_design9',
                        help='select model. can be tex_creator_human_design9, tex_creator_weapon, tex_creator_human_design or other keys in cfg json')
    args = parser.parse_args()

    logging.basicConfig()
    main_server(args.lj_port, args.grpc_cfg_json, args.model_key)
