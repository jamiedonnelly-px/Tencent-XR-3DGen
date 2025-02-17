import os, sys, json, time, traceback
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor, wait
from flask import Flask, request, Response, abort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from run import auto_rig_layer

app_autoRig_layer = Flask(__name__)
cors = CORS(app_autoRig_layer, resources={r"*": {"origins": "*"}})
max_concurrent_users = 16
executor = ThreadPoolExecutor(max_concurrent_users)

def auto_rig_combine(input):
    trajectory_data = input['folder']
    auto_rig_layer(trajectory_data)

@app_autoRig_layer.route("/combine", methods=['POST', 'GET'])
def get_result_combine():
    print("get_result_combine")
    if request.method == 'POST':
        input = request.get_json()
        input_file = input['folder']
    else:
        input_file = request.args['folder']

    print("input: ", input_file)

    try:
        if input_file==None:
            assert(0)
        
        start = time.time()

        future = executor.submit(auto_rig_combine, input)
        
        timeout = 100  # 设置超时时间（以秒为单位）
        done, not_done = wait([future], timeout=timeout)
        
        if future in done:
            end = time.time()
            print('auto rig time: ', end-start)
            result = future.result()
            print(result)
        else:
            raise TimeoutError("The auto-rig process timed out")

    except Exception as e:
        print(e)
        result_error = {'errcode': -1}
        result = json.dumps(result_error, indent=4, ensure_ascii=False)
       
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
      
        abort(Response("Failed!\n" + '\n\r\n'.join('' + line for line in lines)))
    return Response(str(result), mimetype='application/json')

def auto_rig_rig(input):
    pass
    # trajectory_data = input['folder']
    # auto_rig_layer_rig(trajectory_data)

@app_autoRig_layer.route("/auto_rig", methods=['POST', 'GET'])
def get_result_rig():
    print("get_result_rig")
    if request.method == 'POST':
        input = request.get_json()
        input_file = input['folder']
    else:
        input_file = request.args['folder']

    print("input: ", input_file)

    try:
        if input_file==None:
            assert(0)
        
        start = time.time()

        future = executor.submit(auto_rig_rig, input)
        
        timeout = 100  # 设置超时时间（以秒为单位）
        done, not_done = wait([future], timeout=timeout)
        
        if future in done:
            end = time.time()
            print('auto rig time: ', end-start)
            result = future.result()
            print(result)
        else:
            raise TimeoutError("The auto-rig process timed out")

    except Exception as e:
        print(e)
        result_error = {'errcode': -1}
        result = json.dumps(result_error, indent=4, ensure_ascii=False)
       
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
      
        abort(Response("Failed!\n" + '\n\r\n'.join('' + line for line in lines)))
    return Response(str(result), mimetype='application/json')

if __name__ == "__main__":
    app_autoRig_layer.run(host='0.0.0.0', port=8085)