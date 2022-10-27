import sys
import grpc

# import the generated classes
import service.service_spec.tononi_phi_pb2_grpc as grpc_bt_grpc
import service.service_spec.tononi_phi_pb2 as grpc_bt_pb2

from service import registry

TEST_URL = "https://raw.githubusercontent.com/singnet/phi/master/src/input.csv"

if __name__ == "__main__":

    try:
        test_flag = False
        if len(sys.argv) == 2:
            if sys.argv[1] == "auto":
                test_flag = True

        endpoint = input("Endpoint (localhost:{}): ".format(
            registry["tononi_phi_service"]["grpc"])) if not test_flag else ""
        if endpoint == "":
            endpoint = "localhost:{}".format(registry["tononi_phi_service"]["grpc"])

        # Open a gRPC channel
        channel = grpc.insecure_channel("{}".format(endpoint))
        url = input("CSV (URL): ") if not test_flag else TEST_URL
        stub = grpc_bt_grpc.TononiPhiStub(channel)
        request = grpc_bt_pb2.Input(input_url==url,timeout=5)
        response = stub.tononiPhi(request)
        if response.mean == -1 and response.std == -1:
            print("Fail!")
            exit(1)
        print(f"mean = {respponse.mean}")
        print(f"std = {respponse.std}")
        print(f"values = {str(response.values)}")
    except Exception as e:
        print(e)
        exit(1)
