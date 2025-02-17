import time
import grpc
import concurrent.futures
import service_pb2
import service_pb2_grpc
import threading
import uuid
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class MeshService(service_pb2_grpc.MeshServiceServicer):
    def __init__(self, max_run):
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_run)

    def run_func(self, job_id, in_mesh, out_glb):
        with self.jobs_lock:
            self.jobs[job_id] = {}
            self.jobs[job_id]["status"] = "running"
        logging.info(f"begin run {job_id}")
        time.sleep(10)  # Simulate a long-running operation
        with self.jobs_lock:
            self.jobs[job_id]["status"] = "completed"
        return "success"

    def RunFunc(self, request, context):
        job_id = str(uuid.uuid4())
        future = self.executor.submit(self.run_func, job_id, request.in_mesh, request.out_glb)
        logging.info(f"submit done")
        with self.jobs_lock:
            self.jobs[job_id] = {"status": "queued", "future": future}
        result = future.result()
        logging.info(f"run done")
        return service_pb2.RunFuncResponse(job_id=job_id, result=result)

    def QueryRunState(self, request, context):
        with self.jobs_lock:
            job_data = []
            for job_id, job_info in self.jobs.items():
                job_data.append(service_pb2.QueryRunStateResponse.JobData(job_id=job_id, status=job_info["status"]))
        return service_pb2.QueryRunStateResponse(jobs=job_data)

    def CancelJob(self, request, context):
        with self.jobs_lock:
            if request.job_id in self.jobs:
                self.jobs[request.job_id]["future"].cancel()
                self.jobs[request.job_id]["status"] = "cancelled"
                status = "Job cancelled"
            else:
                status = "Job not found"
        return service_pb2.CancelJobResponse(status=status)
    
def serve():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_MeshServiceServicer_to_server(MeshService(max_run=1), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("begin")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()