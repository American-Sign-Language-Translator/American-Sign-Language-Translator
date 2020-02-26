import os
import sys
import numpy as np
from openvino.inference_engine import IENetwork, IECore, IEPlugin, IENetLayer, InferRequest

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Network:

    def __init__(self):
        self.inet = None #IENetwork()
        self.plugin = None #IECore()
        self.input_blob = None #InferRequest.inputs
        self.output_blob = None #InferRequest.outputs
        self.exec_network = None #IECore.load_network
        self.infer_request = None #InferRequest.async_infer

    def layer_check():
        if cpu_extension:
            plugin.add_extension(cpu_extension, "CPU")
        unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        supported_layers = plugin.query_network(network=net, device_name="CPU")
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        
        
    def allinone(input, model):
        inet = Network.net(model)
        exec_net = Network.load_model(inet, input, 'CPU')
        input_blob = next(iter(exec_net.inputs))
        input_layer = inet.inputs[input_blob].shape
        syn_net = Network.inf_(exec_net, input, input_blob)
        dec_net = Network.extract_output(syn_net, exec_net)
        return dec_net
        
        
    def allintwo(input, model, Name_out):
        inet = Network.net(model)
        exec_net = Network.load_model(inet, input, 'CPU')
        input_blob = next(iter(exec_net.inputs))
        input_layer = inet.inputs[input_blob].shape
        syn_net = Network.async_inference(exec_net, input, input_blob, Name_out)
        return syn_net
        
        
    def check_ext(device, cpu_extension):
        plug =self.plugin.add_extension(cpu_extension, device)
        return plug
        
    def net(modelx):
        modelb = os.path.splitext(modelx)[0] + ".bin"
        net = IENetwork(model=modelx, weights=modelb)
        return(net)


    def load_model(self, image, device):
        plugin = IECore()
        exec_net = plugin.load_network(self, device, num_requests=2)
        return exec_net
        
    def inf_(exec_net, image, input_blob):
        res = exec_net.infer({input_blob:image})
        return res

    def async_inference(exec_net, image, input_blob):
        infer_request_handle = exec_net.start_async(request_id=0, inputs={input_blob: image})
        infer_status = infer_request_handle.wait()
        output_blob = next(iter(exec_net.outputs))
        res = exec_net.requests[0].outputs[output_blob]
        return res

    
    def extract_output(self, network):
        output_blob = next(iter(network.outputs))
        det = self[output_blob]
        return det
