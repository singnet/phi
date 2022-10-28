# phi
# 7025

import os
import sys
import time
from concurrent import futures
import random

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), "service_spec"))
import tononi_phi_pb2 as pb2
import tononi_phi_pb2_grpc as pb2_grpc


# SERVICE_API
class ServiceDefinition(pb2_grpc.ServiceDefinitionServicer):
    def tononi_phi(self, request, context):
        output_file = f"/tmp/output_{random.randint(0, 1000000)}.txt"
        command = (
            "python3 ../src/main.py "
            f"--input-url {request.input_url} "
            f"--window-length {request.window_length} "
            f"--timeout {request.timeout} "
            f"--bins {request.bins} "
            f"--nodes {request.nodes} "
            f"--columns-to-skip {request.columns_to_skip} "
            f"--window-start {request.window_start} "
            f"--max-nodes {request.max_nodes} "
            f"--text-output-file {output_file}"
        )
        exit_code = os.system(command)

        if exit_code != 0:
            return pb2.Output()

        with open(output_file, "r") as f:
            std = float(f.readline().strip())
            mean = float(f.readline().strip())
            values = [float(x) for x in f.readline().strip().split(",")]

        return pb2.Output(
            std=std,
            mean=mean,
            values=values,
        )

def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ServiceDefinitionServicer_to_server(ServiceDefinition(), server)
    server.add_insecure_port("[::]:7025")
    server.start()
    print("Server listening on 0.0.0.0:{}".format(7025))
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()
