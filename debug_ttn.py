from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tn_dl.layers.mps_custom import CustomTTN


def safe_shape(tensor):
    if tensor is None:
        return None
    return tuple(tensor.shape)


def summarize_node(node, title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("type(node):", type(node))
    print("name:", getattr(node, "name", None))
    print("axes_names:", getattr(node, "axes_names", None))
    print("shape attr:", getattr(node, "shape", None))

    tensor = getattr(node, "tensor", None)
    print("tensor is None:", tensor is None)
    print("tensor.shape:", safe_shape(tensor))
    if tensor is not None:
        print("tensor.dtype:", tensor.dtype)

    edges = getattr(node, "edges", None)
    if edges is None:
        print("edges: None")
        return

    print("num_edges:", len(edges))
    for i, edge in enumerate(edges):
        edge_name = getattr(edge, "name", None)

        is_dangling = None
        if hasattr(edge, "is_dangling"):
            try:
                is_dangling = edge.is_dangling()
            except TypeError:
                is_dangling = edge.is_dangling

        node1 = getattr(edge, "node1", None)
        node2 = getattr(edge, "node2", None)

        print(
            f"edge[{i}] "
            f"name={edge_name!r} "
            f"dangling={is_dangling} "
            f"node1={getattr(node1, 'name', None)!r} "
            f"node2={getattr(node2, 'name', None)!r}"
        )


def summarize_partner(node, axis_name: str, title: str):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

    partner = node.neighbours(axis_name)
    print("partner type:", type(partner))
    print("partner name:", getattr(partner, "name", None))
    print("partner axes_names:", getattr(partner, "axes_names", None))

    tensor = getattr(partner, "tensor", None)
    print("partner.tensor is None:", tensor is None)
    print("partner.tensor.shape:", safe_shape(tensor))

    return partner


def main() -> None:
    net = CustomTTN(
        sequence_length=3,
        local_dim=2,
        bond_dim=4,
        n_classes=5,
    )

    x = torch.randn(2, 3, 2)

    print("\nLlamando a set_data_nodes() explicitamente...")
    net.set_data_nodes()

    print("\nAnadiendo datos con add_data(x) sin ejecutar contract completo...")
    net.add_data(x)

    block = net.blocks_by_level[0][0]
    left, center, right = block["nodes"]

    summarize_node(left, "LEFT NODE")
    summarize_node(center, "CENTER NODE")
    summarize_node(right, "RIGHT NODE")

    partner_left = summarize_partner(left, "input", "PARTNER LEFT")
    partner_center = summarize_partner(center, "input", "PARTNER CENTER")
    partner_right = summarize_partner(right, "input", "PARTNER RIGHT")

    summarize_node(partner_left, "PARTNER LEFT FULL")
    summarize_node(partner_center, "PARTNER CENTER FULL")
    summarize_node(partner_right, "PARTNER RIGHT FULL")

    print("\nProbando contracciones individuales...")
    try:
        L = left @ partner_left
        summarize_node(L, "RESULTADO L = left @ partner_left")
    except Exception as exc:
        print("Error en L = left @ partner_left:", repr(exc))

    try:
        C = center @ partner_center
        summarize_node(C, "RESULTADO C = center @ partner_center")
    except Exception as exc:
        print("Error en C = center @ partner_center:", repr(exc))

    try:
        R = right @ partner_right
        summarize_node(R, "RESULTADO R = right @ partner_right")
    except Exception as exc:
        print("Error en R = right @ partner_right:", repr(exc))

    print("\nProbando forward completo al final...")
    try:
        root_tensor = net(x)
        print("type(root_tensor):", type(root_tensor))
        print("root_tensor.shape:", safe_shape(root_tensor))
    except Exception as exc:
        print("Error durante net(x):", repr(exc))


if __name__ == "__main__":
    main()
