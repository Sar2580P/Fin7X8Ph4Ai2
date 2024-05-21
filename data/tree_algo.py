## This sturcture makes it easy to operate on a tree and therefore apply the decomposition algorithm
class TreeNode:
    def __init__(self, label):
        self.label = label
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __str__(self, indent=""):
        """
        Recursively constructs the tree structure as a string with indentation.
        """
        result = indent + self.label + "\n"
        for i, child in enumerate(self.children):
            if i == len(self.children) - 1:
                result += child.__str__(indent + "    ")
            else:
                result += child.__str__(indent + "│   ")
        return result

    def get_child_labels(self):
        """
        Returns a list of labels of the children.
        """
        return [child.label for child in self.children]

    def copy(self):
        """
        Creates a deep copy of the tree.
        """
        new_node = TreeNode(self.label)
        for child in self.children:
            new_child = child.copy()
            new_node.add_child(new_child)
        return new_node

    def max_depth(self):
        """
        Recursively calculates the maximum depth of the tree.
        """
        if not self.children:
            return 0
        else:
            return 1 + max(child.max_depth() for child in self.children)

    def no_of_nodes(self):
        """
        Recursively counts all nodes in the tree.
        """
        count = 1  # Include the current node (root)
        for child in self.children:
            count += child.no_of_nodes()
        return count


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
    if not root.children:
        return []
    else:
        def _collect(node, leaf_nodes):
            if not node.children:
                leaf_nodes.append(node)
            else:
                for child in node.children:
                    _collect(child, leaf_nodes)

        leaf_nodes = []
        _collect(root, leaf_nodes)
        return leaf_nodes

"""
# THIS FUNCTION'S LOGIC IS WRONG
# Removes intermidiate node, and connects all the leaves to the root
def decompose_tree(root, inplace=False):
    root = root.copy() if not inplace else root

    leaf_nodes = collect_leaf_nodes(root)

    # Disconnect root from existing children
    root.children = []

    # Connect leaf nodes to root
    for leaf in leaf_nodes:
        root.add_child(leaf)

    return root
#"""


# Connects all the nodes to the root
def decompose_tree(root, inplace=False):
    root = root.copy() if not inplace else root

    # Collect all nodes (except root)
    all_nodes = []
    queue = [root]

    while queue:
        current_node = queue.pop(0)
        all_nodes.append(current_node)
        queue.extend(current_node.children)

    # Disconnect each node from its children
    for node in all_nodes:
        node.children = []

    # Connect root to the collected nodes
    for node in all_nodes[1:]:
        root.add_child(node)

    return root


def collect_boundary_nodes(root, depth, current_depth=0):
    """
    Collects nodes at a specific depth from the root.
    """
    if current_depth == depth:
        return [root]
    elif current_depth < depth:
        if not root.children:
            return [root]
        else:
            nodes = []
            for child in root.children:
                nodes.extend(collect_boundary_nodes(child, depth, current_depth + 1))
            return nodes
    else:
        return []


def decompose_tree_at_boundary_nodes(root, depth, inplace=False):
    """
    Decomposes the tree at boundary nodes
    """
    root = root.copy() if not inplace else root

    boundary_nodes = collect_boundary_nodes(root, depth)
    decomposed_boundary_nodes = []

    for node in boundary_nodes:
        node = decompose_tree(node, inplace=True)
        decomposed_boundary_nodes.append(node)

    return root, decomposed_boundary_nodes


def create_child_to_root_mapping(root):
    """
    Creates a dictionary where keys are child labels and values are root labels.
    """
    child_to_root_mapping = {}
    child_to_root_mapping[root.label] = root.label
    for child in root.children:
        child_to_root_mapping[child.label] = root.label

        #Wedon't want a complex hierarchy but a simple many to one mapping
        #child_to_root_mapping.update(create_child_to_root_mapping(child))

    return child_to_root_mapping


def reverse_mapper(root, depth, include_root=False):
    """
    Creates a dictionary where keys are leaf labels and values are labels of boundary node at given depth.
    """
    reverse_mapping = {}
    if include_root:
        reverse_mapping[root.label] = root.label

    _, boundary_nodes = decompose_tree_at_boundary_nodes(root, depth)

    for node in boundary_nodes:
        child_to_root_mapping = create_child_to_root_mapping(node)
        reverse_mapping.update(child_to_root_mapping)

    return reverse_mapping


def label_reverse_mapping(reverse_mapping, classes):

    labelcode_to_labelname = {row["LabelCode"]: row["LabelName"] for _, row in classes.iterrows()}
    updated_mapping = {}

    for key, value in reverse_mapping.items():

        if key in labelcode_to_labelname:
            key = labelcode_to_labelname[key]

        if value in labelcode_to_labelname:
            value = labelcode_to_labelname[value]

        updated_mapping[key] = value

    return updated_mapping


def replace_labels_in_tree(root, classes, inplace=False):

    def __replace_labels_in_tree(root, labelcode_to_labelname):
        """
        Recursively replaces label codes with label names in the tree.
        """
        if root.label in labelcode_to_labelname:
            root.label = labelcode_to_labelname[root.label]
        for child in root.children:
            __replace_labels_in_tree(child, labelcode_to_labelname)

    root = root.copy() if not inplace else root

    labelcode_to_labelname = {row["LabelCode"]: row["LabelName"] for _, row in classes.iterrows()}

    __replace_labels_in_tree(root, labelcode_to_labelname)

    return root


def class_no_mapper(input_dict):
    # Step 1: Extract all values from the dictionary
    all_values = list(input_dict.values())

    # Step 2: Create a list of distinct values
    distinct_values = list(set(all_values))

    # Step 3: Assign a unique number to each distinct value
    value_to_number = {value: i for i, value in enumerate(distinct_values)}

    return value_to_number


def class_mapper(root=None, depth=None, include_root=False, reverse_mapping=None, class_no_mapping=None):

    if reverse_mapping is None:
        reverse_mapping = reverse_mapper(root, depth, include_root)

    if class_no_mapping is None:
        class_no_mapping = class_no_mapper(reverse_mapping)

    class_mapping = {}
    for key, value in reverse_mapping.items():
        class_mapping[key] = class_no_mapping[value]

    return class_mapping
