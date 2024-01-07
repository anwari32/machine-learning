from graphviz import Digraph
from nn.value import Value

def draw_compute_graph(root: Value, parent_node_id: str=None, dot_instance: Digraph=None, ops_counter=0, graph_name='compute-graph', graph_comment='compute-graph'):
    if dot_instance:
        dot = dot_instance
    else:
        dot = Digraph(graph_name, comment=graph_comment)
    root_id = root.label
    dot.node(name=root_id, label=f"{root_id} | data {root.data:.4f} | grad {root.grad:.4f}", shape="record")
    if parent_node_id:
        dot.edge(root_id, parent_node_id)
    if root.previous:
        ops_counter += 1
        op_node_id = f"{root.operator}{ops_counter}"
        dot.node(name=op_node_id, label=op_node_id)
        dot.edge(op_node_id, root_id)
        for prev in root.previous:
            draw_compute_graph(prev, op_node_id, dot, ops_counter)

    return dot