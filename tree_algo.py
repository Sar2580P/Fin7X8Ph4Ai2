## This sturcture makes it easy to operate on a tree and therefore apply the decomposition algorithm
class TreeNode:
    def __init__(self, label):
        self.label = label
        self.children = []

    def add_child(self, child):
        self.children.append(child)


## To convert JSON to tree format -- pass the dictionary with the root node/label
def json_to_tree(json_data):
    def create_node(data):
        return TreeNode(data["LabelName"])

    def add_children(node, children):
        for child_data in children:
            child_node = create_node(child_data)
            node.add_child(child_node)
            if "Subcategory" in child_data:
                add_children(child_node, child_data["Subcategory"])

    root_node = create_node(json_data)
    if "Subcategory" in json_data:
        add_children(root_node, json_data["Subcategory"])

    return root_node


## To convert tree back to JSON format -- pass the root node
def tree_to_json(root):
    def convert_node(node):
        if not node.children:
            return {"LabelName": node.label}
        else:
            return {
                "LabelName": node.label,
                "Subcategory": [convert_node(child) for child in node.children]
            }

    return convert_node(root)


## Used by the decompostion algorithm
def collect_leaf_nodes(root):
    def _collect(node, leaf_nodes):
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                _collect(child, leaf_nodes)

    leaf_nodes = []
    _collect(root, leaf_nodes)
    return leaf_nodes


## The decomposition algorithm
def decompose_tree(tree):
    root = tree.root
    leaf_nodes = collect_leaf_nodes(root)

    # Disconnect root from existing children
    root.children = []

    # Connect leaf nodes to root
    for leaf in leaf_nodes:
        root.add_child(leaf)

    return root

