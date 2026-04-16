from __future__ import annotations

import math

import tensorkrowch as tk
import torch.nn as nn

from tn_dl.layers.base import BaseTensorNetworkLayer


class CustomTTN(tk.TensorNetwork):
    def __init__(
        self,
        sequence_length: int,
        local_dim: int,
        n_classes: int,
        bond_dim: int,
        out_position: int | None = None,
        boundary: str = "obc",
        name: str = "CustomTTN",
    ) -> None:
        super().__init__(name=name)

        self.sequence_length = sequence_length
        self.local_dim = local_dim
        self.n_classes = n_classes
        self.bond_dim = bond_dim
        self.out_position = out_position
        self.boundary = boundary

        self.input_nodes = []
        self.blocks_by_level = []
        self.root_node = None
        self.classifier_node = None

        self._build_tree()
        self._conenect_levels()
    
    def _build_mps3_block(
        self,
        input_dim: int,
        output_dim: int,
        prefix: str,
    ):
        """
        Aquí contruyo la MPS base.
        """
        left_node = tk.Node(
            shape = (self.bond_dim ,input_dim),
            axes_names = ("right", "input"),
            name = f"{prefix}_left",
            network = self,
        )

        center_node = tk.Node(
            shape = (self.bond_dim, self.bond_dim, input_dim, output_dim),
            axes_names = ("left", "right", "input", "output"),
            name = f"{prefix}_center",
            network = self,
        )
        right_node = tk.Node(
            shape = (self.bond_dim, input_dim),
            axes_names = ("left", "input"),
            name = f"{prefix}_right",
            network = self,
        )

        left_node["right"] ^ center_node["left"]
        center_node["right"] ^ right_node["left"]

        return {
            "nodes": [left_node, center_node, right_node],
            "input_edges": [
                left_node["input"],
                center_node["input"],
                right_node["input"],
            ],
            "output_edge": center_node["output"],
            "output_node": center_node,
        }

    def _build_tree(self) -> None:
        """
        Aqui contruyo la jerarquía de los bloques de MPS de 3 nodos.

        La idea es que no es necesario crear nodos extra para meter los datos, ya que utilizaremos 
        los nodos de las MPS's del primer nivel para que sean los puntos de entrada reales. Eso es 
        posible ya que en cada bloque MPS, los nodos tienen indices "inputs", precisamente para 
        meter los datos en esos índices.

        Aqui se construye la jerarquía pero no se conectan los outputs de un nivel con los inputs
        del siguiente nivel. Eso se hará en `def _connect_levels()`

        """ 
        current_num_sites = self.sequence_length
        current_input_dim = self.local_dim
        level_idx = 0

        while current_num_sites > 1:
            num_blocks = math.ceil(current_num_sites / 3)
            level_blocks = []

            for block_idx in range(num_blocks):
                block = self._build_mps3_block(
                    input_dim = current_input_dim,
                    output_dim = self.bond_dim,
                    prefix = f"level{level_idx}_block{block_idx}",
                )
                level_blocks.append(block)
            
            self.blocks_by_level.append(level_blocks)

            current_num_sites = num_blocks
            current_input_dim = self.bond_dim
            level_idx += 1

    def _conenect_levels(self) -> None:
        """
        Aquí se conectan los niveles. La idea es la siguiente:
            - El nivel 0 recibe los datos reales.
            - El nivel 1 recibe outputs de bloques del nivel 0.
            - El nivel 2 recibe outputs del nivel 1.
            - Etc...
        
        """
        for level_idx in range(len(self.blocks_by_level) - 1):
            current_level = self.blocks_by_level[level_idx]
            next_level = self.blocks_by_level[level_idx + 1]

            outputs = [block["output_edge"] for block in current_level]

            # conectar outputs de 3 en 3 a los inputs del siguiente nivel
            output_ptr = 0
            for next_block in next_level:
                for input_edge in next_block["input_edges"]:
                    if output_ptr < len(outputs):
                        outputs[output_ptr] ^ input_edge
                        output_ptr += 1
                    else:
                        # Decidir como manejar padding/restos
                        pass

    def set_data_nodes(self) -> None:
        """
        Esta función solo aplica al primer nivel. No se usa para conectar bloques entre sí. Solo se
        usa para decirle a TensorKrowch dónde está el batch real.

        Diferencia de conceptos:
        `input_edges`: es el primer nivel, son puertos físicos reales
        `data_nodes`: los crea TensorKrowch cuando llamo a self.network(x)
        """
        input_edges = []

        first_level = self.blocks_by_level[0]
        for block in first_level:
            input_edges.extend(block["input_edges"])

            # Si he hecho padding estructural y tengo más edges que datos reales,
            # aqui es donde puedo cortar a sequence_length
            input_edges = input_edges[:self.sequence_length]

            super().set_data_nodes(input_edges, num_batch_edges = 1)
    
    def contract(self):
        """
        Aquí solo ejecuto las contracciones correspondientes.
        """
        # 1. Contraigo los datos con los bloques del primer nivel
        current_outputs = []

        first_level = self.blocks_by_level[0]
        for block in first_level:
            out_node = self._contract_blocks(block)
            current_outputs.append(out_node)
        
        # 2. Ir subiendo nivel a nivel
        for level_idx in range(1, len(self.blocks_by_level)):
            next_outputs = []
            level_blocks = self.blocks_by_level[level_idx]

            for block_idx, block in enumerate(level_blocks):
                # este bloque ya tiene conectados como inputs
                # los outputs del nivel anterior
                out_node = self._contract_blocks(block)
                next_outputs.append(out_node)
            
            current_outputs = next_outputs
        
        # 3. Debe quedar un único nodo raíz
        root = current_outputs[0]

        # 4. Clasificación final
        logits = self._contract_classifier(root)

        return logits
    
    def _contract_blocks(self, block):
        """
        Esto es un helper para contraer cada bloque.

        Aquí TensorKrowch manda mucho. Este helper debe:
            - Coger los 3 nodos del bloque MPS.
            - Contraerlos internamente.
            - Devolver el nodo resultante con el edge "output" libre si aún debe subir de nivel.
            - Si no sube de nivel/capa más, devuelve el resultado final.
        
        Todavia queda por ajustar:
            - si el edge "output" queda libre como quiero.
            - como se comporta `@` según el tipo de contracción exacta.
            - si necesito `stack` o reordenar edges
        """

        left, center, right = block["nodes"]

        tmp = left @ center
        out = tmp @ right

        return out
    

class MPSClassifier_1(BaseTensorNetworkLayer):
    def __init__(
        self,
        sequence_length: int,
        local_dim: int,
        n_classes: int,
        bound_dim: int,
        out_position: int | None = None,
        boundary: str = "obc",
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.local_dim = local_dim
        self.n_classes = n_classes
        self.bond_dim = bound_dim
        self.out_position = out_position
        self.boundary = boundary

        self.network = CustomTTN(
            sequence_length = sequence_length,
            local_dim = local_dim,
            bond_dim = bound_dim,
            n_classes = n_classes,
            out_position = out_position,
            boundary = boundary,
        )
        self.classifier = nn.Linear(bound_dim, n_classes)
    
    def forward(self, x):
        root_vec = self.network(x)
        return self.classifier(root_vec)
