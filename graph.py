import matplotlib
matplotlib.use('Agg')
from graphviz import Digraph
import torch
from torch.autograd import Variable

first = True

def make_dot(var, params, img_size):
    """ Produces Graphviz representation of PyTorch autograd graph
    
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    # print(param_map)
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="42,42"),engine='dot')
    seen = set()
    
    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    # dot.node("img", "image" + size_to_str(img_size), fillcolor='orange')

    def add_nodes(var):
        global first
        if var not in seen:
            if torch.is_tensor(var):
                node_name = 'a%s' % (size_to_str(var.size()))
                dot.node(str(id(var)), node_name, fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                # node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                node_name = 'w%s' % (size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                node_name = str(type(var).__name__)
                node_name = node_name.replace('Backward', '')
                node_name = node_name.replace('Threshold', 'ReLU')
                node_name = node_name.replace('Addmm', 'FC')
                node_name = node_name.replace('DivConstant', 'Softmax')
                dot.node(str(id(var)), node_name)
                if first:
                    # dot.edge("img", str(id(var)))
                    first = False
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    # print(type(var))
    add_nodes(var.grad_fn)

    # dot.node("a", "activations", fillcolor='orange')
    # dot.node("w", "weights", fillcolor='lightblue')
    # dot.node("l", "operations") 

    return dot

# from torchvision import models
from arguments import get_arguments
from combine import Combine
from models.vgg import *
from models.dpn import *
from models.lenet import *
from models.senet import *
from models.resnet import *
from models.resnext import *
from models.densenet import *
from models.googlenet import *
from models.mobilenet import *
from models.shufflenet import *
from models.preact_resnet import *
from models.smallnet import *

parser = get_arguments()
opt = parser.parse_args()
img_size =(1,3,32,32) 
inputs = torch.randn(1,3,32,32)
# resnet18 = models.resnet18()
# model = VGG16()
arch_config = {}
arch_config['num_classes'] = 10
arch_config['num_channels'] = 3
if opt.E > 1: 
    arch_config['balance'] = opt.balance
else:
    arch_config['balance'] = False
model = Combine(opt.arch, opt.E, opt.probs, ensemble=True, **arch_config)
# model = SmallNetOriginal()
y,_ = model(Variable(inputs))
# print(y)
print(model)
g = make_dot(y, model.state_dict(), img_size)
# g.view()

g.render('./model_digraphs/' + opt.arch + "_#Branches:" + str(opt.E) + "_SQRT:"+str(opt.balance))
