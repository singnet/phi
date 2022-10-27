# phi
# 7025
import os
import sys

import grpc

sys.path.append(os.path.join(os.path.dirname(__file__), "service_spec"))
import tononi_phi_pb2 as pb2
import tononi_phi_pb2_grpc as pb2_grpc


# TEST_CODE
def call_implemented_code(channel):
    # Check the compiled proto file (.py) to get method names
    stub = pb2_grpc.ServiceDefinitionStub(channel)
    input_url = "https://raw.githubusercontent.com/singnet/phi/master/Sophia_Meditation.csv"
    input_ = pb2.Input(
        input_url=input_url,
        window_length=20,
        timeout=1,
        bins=5,
        nodes=10,
        columns_to_skip=0,
        window_start=1,
        max_nodes=50,
    )
    response = stub.tononi_phi(input_)
    return response


def main():
    # Connect to the server
    with grpc.insecure_channel("localhost:7025") as channel:
        # Call TEST_CODE
        output = call_implemented_code(channel)
        print(output)


if __name__ == "__main__":
    main()
