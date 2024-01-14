from graphviz import Digraph
from nn.value import Value
from nn.activation_functions import dtanh

def draw_compute_graph(root: Value, parent_node_id: str=None, dot_instance: Digraph=None, ops_counter=0, graph_name='compute-graph', graph_comment='compute-graph'):
    """Draw compute graph"""
    ops_counter += 1
    if dot_instance:
        dot = dot_instance
    else:
        dot = Digraph(graph_name, comment=graph_comment, graph_attr={'rankdir': 'LR'})
    root_id = root.label
    dot.node(name=root_id, label=f"{root_id} | data {root.data:.4f} | grad {root.grad:.4f}", shape="record")
    if parent_node_id:
        dot.edge(root_id, parent_node_id)
    if root.previous:
        op_node_id = f"{root.operator}{root.compute_order}"
        dot.node(name=op_node_id, label=op_node_id)
        dot.edge(op_node_id, root_id)
        for prev in root.previous:
            draw_compute_graph(prev, op_node_id, dot, ops_counter=ops_counter)
    return dot


def backprop(root, grad_accumulator=None):
    """Backpropagation"""
    if grad_accumulator == None:
        root.grad = 1
    grad_accumulator = root.grad
    if root.operator:
        opname = root.operator[0:3]
        if opname == "add":
            for prev in root.previous:
                prev.grad = grad_accumulator
                backprop(prev, grad_accumulator)
        elif opname == "mul":
            root.previous[0].grad = grad_accumulator * root.previous[1].data
            backprop(root.previous[0], grad_accumulator * root.previous[0].grad)

            root.previous[1].grad = grad_accumulator * root.previous[0].data
            backprop(root.previous[1], grad_accumulator * root.previous[1].grad)
        elif root.operator[0:4] == "tanh":
            for prev in root.previous:
                prev.grad = dtanh(prev.data) * grad_accumulator
                backprop(prev, grad_accumulator * prev.grad)
    return root    

def update_value_with_grad(root, step=1):
    """Update node value with its gradient"""
    root.data = root.data * step * root.grad
    root.grad = 0
    if root.operator:
        for prev in root.previous:
            update_value_with_grad(prev)
    return root

