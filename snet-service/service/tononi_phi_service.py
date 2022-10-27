import os
import sys
import logging

import grpc
import concurrent.futures as futures

import service.common

# Importing the generated codes from buildproto.sh
import service.service_spec.tononi_phi_pb2_grpc as grpc_bt_grpc
from service.service_spec.tononi_phi_pb2 import Output

import service.tononi_phi as tp

logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger("tononi_phi_service")

# Create a class to be added to the gRPC server
# derived from the protobuf codes.
class TononiPhiServicer(grpc_bt_grpc.TononiPhiServicer):
    def __init__(self):
        # Just for debugging purpose.
        log.debug("TononiPhiServicer created")

    @staticmethod
    def tononiPhi(request, context):
        try:
            os.system(f"{os.getenv('PROJECT_DIR')}/src/main.py " + \
                f"--input-url {request.input_url} " + \
                f"--output-file {request.input_url} " + \
                f"--window-length {request.window_length} " + \
                f"--timeout {request.timeout} " + \
                f"--bins {request.bins} " + \
                f"--nodes {request.nodes} " + \
                f"--columns-to-skip {request.columns_to_skip} " + \
                f"--window-start {request.window_start} " + \
                f"--max-nodes {request.max_nodes} ")
            with open("/tmp/output.txt") as f:
                mean = float(f.readline())
                std = float(f.readline())
                values = [float(v) for v in f.readline().split(",")]
        except Exception e:
            error_msg = str(e)
            log.error(error_msg)
            context.set_details(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            return Output(mean=-1, std=-1)
        else:
            return Output(mean=mean, std=std, values=values)


# The gRPC serve function.
#
# Params:
# max_workers: pool of threads to execute calls asynchronously
# port: gRPC server port
#
# Add all your classes to the server here.
# (from generated .py files by protobuf compiler)
def serve(max_workers=1, port=7777):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=[
        ('grpc.max_send_message_length', 25 * 1024 * 1024),
        ('grpc.max_receive_message_length', 25 * 1024 * 1024)])
    grpc_bt_grpc.add_TononiPhiServicer_to_server(TononiPhiServicer(), server)
    server.add_insecure_port("[::]:{}".format(port))
    return server


if __name__ == "__main__":
    """
    Runs the gRPC server to communicate with the SNET Daemon.
    """
    parser = service.common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    service.common.main_loop(serve, args)
