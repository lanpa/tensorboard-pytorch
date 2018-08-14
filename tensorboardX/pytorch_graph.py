import time
import warnings

from distutils.version import LooseVersion

from .proto.attr_value_pb2 import AttrValue
from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.step_stats_pb2 import RunMetadata, StepStats, DeviceStepStats, NodeExecStats, AllocatorMemoryUsed
from .proto.tensor_shape_pb2 import TensorShapeProto
from .proto.versions_pb2 import VersionDef


def parse(graph):
    import torch
    scope = {}
    for n in graph.nodes():
        inputs = [i.uniqueName() for i in n.inputs()]
        for i in range(1, len(inputs)):
            if inputs[i] not in scope.keys():
                scope[inputs[i]] = n.scopeName()

        uname = next(iter(n.outputs())).uniqueName()
        assert n.scopeName() != '', '{} has empty scope name'.format(n)
        scope[uname] = n.scopeName()
    if LooseVersion(torch.__version__) >= LooseVersion("0.4"):
        scope['0'] = 'input'
    else:
        scope['1'] = 'input'

    nodes = []

    for count, n in enumerate(graph.outputs()):
        uname = 'output' + str(count)
        scope[uname] = 'output'
        nodes.append({'name': uname, 'op': 'output', 'inputs': [
                     n.uniqueName()], 'attr': 'output'})

    for n in graph.nodes():
        try:
            attrs = str({k: n[k] for k in n.attributeNames()})
        except RuntimeError as e:
            attrs = str(n).strip()
            warnings.warn(
                "Error getting attributes of node {}, error is {}".format(
                    attrs, e))
        # singlequote will be escaped by tensorboard
        attrs = attrs.replace("'", ' ')
        inputs = [i.uniqueName() for i in n.inputs()]
        # FIXME: only first output is considered (only Dropout)
        outputnode = next(iter(n.outputs()))
        uname = outputnode.uniqueName()
        if outputnode.type().kind() == 'TensorType':
            outputsize = outputnode.type().sizes()
            nodes.append({'name': uname,
                          'op': n.kind(),
                          'inputs': inputs,
                          'attr': attrs,
                          'outputsize': outputsize})
        else:
            nodes.append({'name': uname, 'op': n.kind(),
                          'inputs': inputs, 'attr': attrs})

    for n in graph.inputs():
        uname = n.uniqueName()
        if uname not in scope.keys():
            scope[uname] = 'unused'
        outputsize = n.type().sizes()
        nodes.append({'name': uname,
                      'op': 'Parameter',
                      'inputs': [],
                      'attr': str(n.type()),
                      'outputsize': outputsize})

    mapping = {}
    for n in nodes:
        mapping[n['name']] = scope[n['name']] + '/' + \
            n['op'].replace('onnx::', '') + '_' + n['name']
    for n in nodes:
        n['name'] = mapping[n['name']]
        for i, s in enumerate(n['inputs']):
            n['inputs'][i] = mapping[s]
    return nodes


def graph(model, args, verbose=False):
    import torch
    with torch.onnx.set_training(model, False):
        try:
            trace, _ = torch.jit.get_trace_graph(model, args)
        except RuntimeError:
            print('Error occurs, No graph saved')
            _ = model(args)  # don't catch, just print the error message
            print("Checking if it's onnx problem...")
            try:
                import tempfile
                torch.onnx.export(
                    model, args, tempfile.TemporaryFile(), verbose=True)
            except RuntimeError:
                print("Your model fails onnx too, please report to onnx team")
            return GraphDef(versions=VersionDef(producer=22))
    if LooseVersion(torch.__version__) >= LooseVersion("0.4.1"):
        torch.onnx._optimize_trace(
            trace, torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    elif LooseVersion(torch.__version__) >= LooseVersion("0.4"):
        torch.onnx._optimize_trace(trace, False)
    else:
        torch.onnx._optimize_trace(trace)
    graph = trace.graph()
    if verbose:
        print(graph)
    list_of_nodes = parse(graph)
    nodes = []
    node_stats = []
    for node in list_of_nodes:
        if 'outputsize' in node.keys():
            shapeproto = TensorShapeProto(
                dim=[TensorShapeProto.Dim(size=d) for d in node['outputsize']])
            nodes.append(
                NodeDef(name=node['name'], op=node['op'], input=node['inputs'],
                        attr={'lanpa': AttrValue(s=node['attr'].encode(encoding='utf_8')),
                              '_output_shapes': AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))}))
            # FIXME: fill with profile data
            node_stats.append(NodeExecStats(node_name=node['name'],
                                            all_start_micros=int(
                                                time.time() * 1e7),
                                            all_end_rel_micros=42,
                                            memory=[AllocatorMemoryUsed(allocator_name="cpu",
                                                                        total_bytes=19950829,
                                                                        peak_bytes=19950829,
                                                                        live_bytes=19950829)]))
        else:
            nodes.append(
                NodeDef(name=node['name'], op=node['op'], input=node['inputs'],
                        attr={'lanpa': AttrValue(s=node['attr'].encode(encoding='utf_8'))}))

    stepstats = RunMetadata(step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0",
                                                                            node_stats=node_stats)]))
    return GraphDef(node=nodes, versions=VersionDef(producer=22)), stepstats
