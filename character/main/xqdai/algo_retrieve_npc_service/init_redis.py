import redis
import sys

def connect_redis(host="",password="",port=6379,db_id=0):
    while True:
        try:
            r = redis.Redis(
                host=host,
                password=password,
                port=port,
                db=db_id,
                health_check_interval=30,
            )
            # r = redis.Redis(host='localhost', port=6379, db=0,health_check_interval=30)
            r.ping()
            return r
        except redis.exceptions.ConnectionError as e:
            print("连接失败，尝试重新连接...")
            time.sleep(5)


def init(db_id=0):
    host=""
    password=""
    port=6379
    redis_conn = connect_redis(host=host,password=password,port=port,db_id=db_id)

    redis_conn.delete("npc_retrieve_redis_lock")
     
if __name__ == '__main__':
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] 
    db_id = argv[0]
    print(db_id)
    init(db_id)
