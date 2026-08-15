from __future__ import annotations

from typing import Optional, Literal
from abc import ABC, abstractmethod
import bs4
import os
import re
import logging
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import networkx as nx
import tiktoken

from agentlab.llm.chat_api import RetryError


tokenizer = tiktoken.encoding_for_model("gpt-4o")

TreeType = Literal["axtree", "pruned_html", "html"]


class Tree(ABC):
    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Returns the node with the given ID."""
        pass

    @abstractmethod
    def replace_node(self, node_id: str, new_node: Node):
        """Replaces a node with a new node in the tree."""
        pass

    @abstractmethod
    def get_collapsed_nodes(self) -> list[Node]:
        """Returns a list of nodes that have been collapsed."""
        pass


class Node:
    def __init__(
        self,
        node_id: str = None,
        role: str = None,
        name: str = None,
        attributes: dict = None,
        n_tokens: int = None,
        line: str = None,
        children: list[Node] = None,
        line_number: int = None,
    ):
        self.node_id = node_id  # ID like [47], if available
        self.role = role  # Role like 'generic', 'button', etc.
        self.name = name  # Accessible name or description
        self.attributes = attributes if attributes else {}  # Attributes like live='polite'
        self.n_tokens = n_tokens  # Number of tokens in the node
        self.line = line  # Original line from the input string
        self.children: list[Node] = children if children else []  # List of child nodes
        self.line_number = line_number  # Add line number attribute

    def add_child(self, child_node):
        """Adds a child node to the current node."""
        self.children.append(child_node)

    def get_nb_nodes_in_subtree(self) -> int:
        """Returns the total number of nodes in the tree."""
        n_nodes = 1
        for child in self.children:
            n_nodes += child.get_nb_nodes_in_subtree()
        return n_nodes

    def display(self, level=0):
        """Recursively prints the tree with indentation to represent hierarchy."""
        indent = "  " * level
        attributes_str = ", ".join([f"{k}: {v}" for k, v in self.attributes.items()])
        if self.line_number is not None:
            print(
                f"{self.line_number} {indent}- ID: {self.node_id}, Role: {self.role}, Name: '{self.name}', Attributes: {{{attributes_str}}}, Nb tokens: {self.n_tokens}"
            )
        else:
            print(
                f"{indent}- ID: {self.node_id}, Role: {self.role}, Name: '{self.name}', Attributes: {{{attributes_str}}}, Nb tokens: {self.n_tokens}"
            )

        # Recursively display each child node
        for child in self.children:
            child.display(level + 1)

    def display_axtree(self, level=0):
        """Recursively prints the AXTree with indentation to represent hierarchy."""
        indent = "  " * level
        if self.line_number is not None:
            print(f"{self.line_number} {indent}{self.line}")
        else:
            print(f"{indent}{self.line}")
        # Recursively display each child node
        for child in self.children:
            child.display_axtree(level + 1)

    def to_string(self) -> str:  # TODO: Rename to __str__() ?
        """Recursively builds string AXTree with indentation to represent hierarchy."""
        if self.line_number is not None:
            result = f"{self.line_number} {self.line}\n"
        else:
            result = f"{self.line}\n"
        # Recursively build string representation of each child node
        for child in self.children:
            result += child.to_string()
        return result

    def get_n_tokens_subtree(self) -> int:
        """Returns the total number of tokens in the subtree with root Node."""
        if self.n_tokens is None:
            self.n_tokens = len(tokenizer.encode(self.line))

        if len(self.children) == 0:
            return self.n_tokens

        subtree_n_tokens = self.n_tokens

        for child in self.children:
            subtree_n_tokens += child.get_n_tokens_subtree()

        return subtree_n_tokens

    def preorder_traversal(self):
        """Performs a preorder traversal of the tree and returns a list of nodes."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.preorder_traversal())
        return nodes


class AxTree(Tree):
    def __init__(self, tree_string, add_line_numbers: bool = False):
        """Initializes the tree by parsing a string representation of the accessibility tree."""
        self.root = self._parse_tree(tree_string, add_line_numbers)
        self.id_index = self._build_id_index()
        self.line_number_index = self._build_line_number_index()

    def __len__(self):
        """Returns the number of nodes in the tree."""
        return len(self.root.preorder_traversal())

    def _build_id_index(self) -> dict[str, Node]:
        """Builds an index of node IDs to node objects."""
        id_index = {}

        def build_id_index_recursive(node: Node):
            if node.node_id:
                id_index[node.node_id] = node
            for child in node.children:
                build_id_index_recursive(child)

        build_id_index_recursive(self.root)
        return id_index

    def _build_line_number_index(self) -> dict[int, Node]:
        """Builds an index of line numbers to node objects."""
        line_number_index = {}

        def build_line_number_index_recursive(node: Node):
            if node.line_number is not None:
                line_number_index[node.line_number] = node
            for child in node.children:
                build_line_number_index_recursive(child)

        build_line_number_index_recursive(self.root)
        return line_number_index

    def _parse_tree(self, tree_string: str, add_line_numbers: bool) -> Node:
        """Parses the input string and constructs the accessibility tree."""
        lines = tree_string.rstrip().splitlines()
        stack: list[Node] = []  # Stack to keep track of the parent nodes
        root = None
        line_number = 0  # Initialize line number counter

        for line in lines:
            line_number += 1  # Increment line number for each processed line
            original_line = line
            stripped_line = line.lstrip()
            level = len(line) - len(stripped_line)

            # Normalize levels to multiples of 4 spaces or tabs (1 tab == 4 spaces)
            level = level // 4 if " " in line[:level] else level // 1

            node_id, role, name, attributes, n_tokens = self._parse_line(stripped_line)
            if level == 0 and node_id == None:
                node_id = "Root"
            node = Node(node_id, role, name, attributes, n_tokens, original_line)
            # Add line number to node attributes
            if add_line_numbers:
                node.line_number = line_number

            if not stack:
                root = node
            else:
                while len(stack) > level:
                    stack.pop()

                parent_node = stack[-1]
                parent_node.add_child(node)

            stack.append(node)

        return root

    def _parse_line(self, line: str) -> tuple[str, str, str, dict, int, str]:
        """Parses a line to extract node ID, role, name, and attributes."""
        id_match = re.match(r"\[(.*?)\]", line)
        node_id = id_match.group(1) if id_match else None
        line = re.sub(r"^\[.*?\]\s*", "", line)

        role_match = re.match(r"(\w+)", line)
        role = role_match.group(1) if role_match else "unknown"

        name_match = re.search(r"'(.*?)'", line)
        name = name_match.group(1) if name_match else None

        attributes = {}
        attributes_part = line.split("'")[-1]
        attr_matches = re.findall(r"(\w+)=['\"]?(.*?)['\", ]", attributes_part)
        for key, value in attr_matches:
            attributes[key] = value

        n_tokens = len(tokenizer.encode(line))
        return node_id, role, name, attributes, n_tokens

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Returns the node with the given ID."""
        return self.id_index.get(node_id, None)

    def get_node_by_line_number(self, line_number: int) -> Optional[Node]:
        """
        Returns the node with the given line number.

        Args:
            line_number: The line number to search for

        Returns:
            Node: The node with the specified line number, or None if not found
        """
        return self.line_number_index.get(line_number, None)

    def replace_node(self, node_id: str, new_node: Node):
        """Replaces a node with a new node in the tree."""

        def replace_node_recursive(node: Node) -> Node:
            if node.node_id == node_id:
                return new_node
            for i, child in enumerate(node.children):
                node.children[i] = replace_node_recursive(child)
            return node

        self.root = replace_node_recursive(self.root)

    def get_collapsed_nodes(self) -> list[tuple[Node, int]]:
        """Returns a list of tuples containing collapsed nodes and their levels."""
        collapsed_nodes = []

        def get_collapsed_nodes_recursive(node: Node, level: int):
            for child in node.children:
                if "collapsed" in child.line and node not in [n for n, _ in collapsed_nodes]:
                    collapsed_nodes.append((node, level))
                get_collapsed_nodes_recursive(child, level + 1)

        get_collapsed_nodes_recursive(self.root, 0)
        return collapsed_nodes

    def get_collapsed_nodes_ids(self) -> list[Node]:
        """Returns a list of nodes that have been collapsed."""
        collapsed_nodes = []

        def get_collapsed_nodes_recursive(node: Node):
            for child in node.children:
                if "collapsed" in child.line and node not in collapsed_nodes:
                    collapsed_nodes.append(node.node_id)
                get_collapsed_nodes_recursive(child)

        get_collapsed_nodes_recursive(self.root)
        return list(set(collapsed_nodes))

    def summarize_and_replace(
        self,
        original_tree: AxTree,
        sub_tree: Node,
        summarizer: callable,
    ):
        nb_lines = sub_tree.get_nb_nodes_in_subtree() - 1
        n_tabs = sub_tree.line.rstrip().count("\t") + 1  # Shift to the right
        # Summarize the subtree with a LLM
        summary = summarizer(sub_tree.to_string())

        # Build a new node with new children being the summary and info about nb lines collapsed
        summarized_tree_object = Node(
            node_id=sub_tree.node_id,
            name=sub_tree.name,
            attributes=sub_tree.attributes,
            line=f"{sub_tree.line}",
            children=[
                Node(
                    line="{tabs}<collapsed {nb_lines} lines>".format(
                        tabs=(n_tabs) * "\t", nb_lines=nb_lines
                    )
                ),
                Node(line="{tabs}{summary}".format(tabs=(n_tabs + 1) * "\t", summary=summary)),
                Node(line="{tabs}</collapsed>".format(tabs=(n_tabs) * "\t")),
            ],
        )

        # Replace the subtree with the summarized tree
        original_tree.replace_node(sub_tree.node_id, summarized_tree_object)
        return original_tree, summarized_tree_object

    def prune_children_except_path(
        self, node, trim_level, current_level, node_at_trim_level, path_to_target
    ):
        if current_level == trim_level:
            if node != node_at_trim_level:
                node.children = []
        for child in node.children:
            self.prune_children_except_path(
                child, trim_level, current_level + 1, node_at_trim_level, path_to_target
            )

    def trim_tree(self, target_id, k):
        """
        Trims the tree by removing all nodes except the path from the target node to the node on k levels above it.

        Input:
        - target_id: ID of the node to be trimmed
        - k: Number of levels to go above the target node before trimming

        Output:
        - Returns the root of the trimmed tree
        """
        path_to_target: list[Node] = []
        print("target_id", target_id)
        print("k", k)

        def find_and_record_path(node, target_id):
            if node.node_id == target_id:
                path_to_target.append(node)
                return True
            for child in node.children:
                if find_and_record_path(child, target_id):
                    path_to_target.append(node)
                    return True
            return False

        if not find_and_record_path(self.root, target_id):
            print(f"Node with ID {target_id} not found.")
            return None

        print("Path to target:", end=" ")
        for node in path_to_target:
            print(node.node_id, end=", ")
        print()

        trim_level = max(0, len(path_to_target) - k - 1)
        # reverse the path_to_target
        path_to_target = path_to_target[::-1]
        node_at_trim_level = path_to_target[trim_level]
        self.prune_children_except_path(
            self.root, trim_level, 0, node_at_trim_level, path_to_target
        )

        return self.root

    def chunk_tree(self, max_n_descendants: int, min_n_tokens_of_chunk: int) -> list[Node]:
        """
        Traverses the tree and creates chunks where a node has fewer than m descendants.
        Returns a list of chunks. (The maximal chunk that is smaller than m nodes)

        Args:
            max_n_descendants: Maximum number of descendants allowed in a chunk

        Returns:
            list[Node]: A list of chunks (subtrees) that have fewer than m descendants
        """

        if max_n_descendants < 1:
            raise ValueError(
                "m must be at least 1 because it is the size of a subtree and no subtree can be empty."
            )

        chunks = []

        def count_and_chunk(node: Node):
            # Recursively count descendants
            num_descendants = 1  # Include the current node
            total_tokens = node.get_n_tokens_subtree()
            children_status = []
            # print(f"Processing node {node.node_id}: {len(node.children)} children.")
            for child in node.children:
                n_children, status, total_tokens = count_and_chunk(child)
                # print(f"Processing child {child.node_id}: {n_children} descendants, status: {status}")
                num_descendants += n_children
                children_status.append(status)

            # Create a chunk if the number of descendants is less than m
            if num_descendants <= max_n_descendants:
                status = "Chunked"
            else:
                status = "Not Chunked"

            if status == "Not Chunked":
                for i in range(len(children_status)):
                    if children_status[i] == "Chunked":
                        child = node.children[i]
                        # print(
                        #     f"Creating chunk for node {child.node_id} with father of {num_descendants} descendants."
                        # )
                        if (
                            child.get_n_tokens_subtree() > min_n_tokens_of_chunk
                            and child.get_nb_nodes_in_subtree() > 1
                            and child.node_id != None
                        ):
                            chunks.append(child)

            return num_descendants, status, total_tokens

        count_and_chunk(self.root)
        if chunks == []:
            chunks.append(self.root)

        return chunks

    def chunk_to_string(self, chunk_root):
        """
        Converts a subtree (chunk) rooted at chunk_root into a string representation.
        """
        result = []

        def dfs(node, level=0):
            indent = "  " * level
            result.append(f"{indent}- ID: {node.node_id}, Role: {node.role}, Name: '{node.name}'")
            for child in node.children:
                dfs(child, level + 1)

        dfs(chunk_root)
        return "\n".join(result)

    def get_tree_size_and_depth(self, root=None):
        """
        Returns the size and depth of the tree rooted at the given node.
        """
        if root is None:
            root = self.root
        if not root.children:
            return 1, 1
        sizes_and_depths = [self.get_tree_size_and_depth(child) for child in root.children]
        size = 1 + sum(size for size, _ in sizes_and_depths)
        depth = 1 + max(depth for _, depth in sizes_and_depths)
        return size, depth
