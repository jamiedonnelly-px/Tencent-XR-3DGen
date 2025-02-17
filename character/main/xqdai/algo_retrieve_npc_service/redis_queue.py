import redis
from app import app

REDIS_PASSWORD = app.config.get("REDIS_PASSWORD")
REDIS_HOSTNAME = app.config.get("REDIS_HOSTNAME")
CELERY_BROKER_URL = app.config.get("CELERY_BROKER_URL")
redis_client = redis.Redis(
    host=REDIS_HOSTNAME,
    port=6379,
    db=0,
    password=REDIS_PASSWORD,
    charset="UTF-8",
    encoding="UTF-8",
)
retries = 10


def add_job_2_queue(job_queue, job_id):
    i = 0
    while i < retries:  # 最多只尝试添加任务[self.retries]次
        try:
            redis_client.watch(job_queue)
            pipe = redis_client.pipeline()
            pipe.rpush(job_queue, job_id)
            pipe.execute()
            return
        except redis.WatchError:
            continue

    if i == retries:
        raise redis.WatchError(f"向队列添加任务失败 {job_id}")


def remove_job_from_queue(job_queue, job_id):
    i = 0
    while i < retries:  # 最多只尝试移除任务[self.retries]次
        try:
            redis_client.watch(job_queue)
            pipe = redis_client.pipeline()
            pipe.lrem(job_queue, 1, job_id)
            pipe.execute()
            return
        except redis.WatchError:
            continue

    if i == retries:
        raise redis.WatchError(f"从队列移除任务失败 {job_id}")
