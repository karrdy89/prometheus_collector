import json
from multiprocessing.pool import ThreadPool
from typing import Union

import xmltodict
from fastapi import FastAPI, APIRouter, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CollectorRegistry, Gauge, generate_latest
import subprocess

registry = CollectorRegistry()

app = FastAPI(debug=False)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])
router = APIRouter()

metrics: dict[str, Union[dict, Gauge]] = {"gpu_temperature": Gauge(
                            f"gpu_temperature",
                            f"temperature(C) of gpu",
                            registry=registry)}


def run_command(command: list[str]) -> str:
    p = subprocess.run(command,
                       capture_output=True, text=True)
    p = p.stdout.strip()
    return p


@router.get("/metrics")
def get_metrics():
    cmd_docker_stats = ["docker",
                        'stats',
                        '--no-stream',
                        '--format',
                        "{\"container\": \"{{ .Container }}\", \"name\": \"{{ .Name }}\", \"memory\": \"{{ .MemPerc }}\", \"cpu\": \"{{ .CPUPerc }}\"}"]
    cmd_gpu_stats = ["nvidia-smi",
                     "-x",
                     "-q"]
    commands = [cmd_docker_stats, cmd_gpu_stats]
    with ThreadPool() as pool:
        for idx, result in enumerate(pool.map(run_command, commands)):
            if idx == 0:
                for line in result.split('\n'):
                    line = json.loads(line)
                    container_name = line.get("name")
                    if container_name not in metrics:
                        metrics[container_name] = {}
                        metrics[container_name]["cpu"] = Gauge(
                            f"container_usage_cpu_{container_name}",
                            f"cpu_usage(%) of {container_name} container",
                            registry=registry)
                        metrics[container_name]["memory"] = Gauge(
                            f"container_usage_memory_{container_name}",
                            f"memory_usage(%) of {container_name} container",
                            registry=registry)
                    cpu_usage = float(line.get("cpu")[:-1])
                    metrics.get(container_name).get("cpu").set(cpu_usage)
                    mem_usage = float(line.get("memory")[:-1])
                    metrics.get(container_name).get("memory").set(mem_usage)
            else:
                gpu_stats = xmltodict.parse(result)
                gpu_stats = gpu_stats.get("nvidia_smi_log").get("gpu")
                gpu_temperature = gpu_stats.get("temperature")
                gpu_current_temp = float(gpu_temperature.get("gpu_temp").split(" ")[0])
                metrics.get("gpu_temperature").set(gpu_current_temp)

    return Response(generate_latest(registry=registry),
                    media_type="text/plain")


app.include_router(router)
