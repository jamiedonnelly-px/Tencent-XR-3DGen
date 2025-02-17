import grpc
import service_pb2
import service_pb2_grpc
import concurrent.futures
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_func(stub, in_mesh, out_glb):
    response = stub.RunFunc(service_pb2.RunFuncRequest(in_mesh=in_mesh, out_glb=out_glb))
    print(f"RunFunc result: {response.result}, job_id: {response.job_id}")

def query_run_state(stub):
    response = stub.QueryRunState(service_pb2.QueryRunStateRequest())
    print("QueryRunState:")
    for job_data in response.jobs:
        logging.info(f"Job ID: {job_data.job_id}, Status: {job_data.status}")
        

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = service_pb2_grpc.MeshServiceStub(channel)

        # print('once begin')
        # run_func(stub, "in_mesh", "out_glb")
        # print('once done')
        
        # Run multiple tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_func, stub, "in_mesh", "out_glb") for _ in range(5)]

            # Query the run state every second
            while not all(f.done() for f in futures):
                query_run_state(stub)
                time.sleep(1)
                
            # Wait for all run_func tasks to complete
            concurrent.futures.wait(futures)

        # Query the run state after all tasks are completed
        query_run_state(stub)

if __name__ == '__main__':
    run()