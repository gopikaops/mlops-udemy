import grpc
import protos.cats_vs_dogs_pb2_grpc as cats_vs_dogs_pb2_grpc
import protos.cats_vs_dogs_pb2 as cats_vs_dogs_pb2
from concurrent import futures
import logging
from models import Models

class CatsVsDogsService(cats_vs_dogs_pb2_grpc.CatsVsDogsServiceServicer):
    def CatsVsDogsTensorflowInference(self, request, context):
        try:
            logging.info("Running Tensorflow gRPC inference...")
            img_array = models.load_image_tf(request.image)
            result = models.predict_tensorflow(img_array)
            return cats_vs_dogs_pb2.CatsVsDogsResponse(cls=result['class'])
        except Exception as e:
            message = "Server error while processing image!"
            logging.error(f"{message} {e}", exc_info=True)
            server_error = grpc.RpcError(message)
            server_error.code = lambda: grpc.StatusCode.INTERNAL
            raise server_error
    
    def CatsVsDogsPyTorchInference(self, request, context):
        try:
            logging.info("Running PyTorch gRPC inference...")
            img_tensor = models.load_image_pytorch(request.image)
            result = models.predict_pytorch(img_tensor)
            return cats_vs_dogs_pb2.CatsVsDogsResponse(cls=result['class'])
        except Exception as e:
            message = "Server error while processing image!"
            logging.error(f"{message} {e}", exc_info=True)
            server_error = grpc.RpcError(message)
            server_error.code = lambda: grpc.StatusCode.INTERNAL
            raise server_error 

def serve():
    server = grpc.server(futures.ThreadPoolExecutor())
    cats_vs_dogs_pb2_grpc.add_CatsVsDogsServiceServicer_to_server(CatsVsDogsService(), server)
    server.add_insecure_port("[::]:50000")
    server.start()
    logging.info("Started gRPC server on localhost:50000...")
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    models = Models()
    serve()