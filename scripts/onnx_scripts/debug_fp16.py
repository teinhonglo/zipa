
import onnx
from onnx import helper

model_path = "checkpoints/zipa-t-small-300k/exp/encoder-epoch-999-avg-1.fp16.onnx"
model = onnx.load(model_path)

print(f"Inspecting model: {model_path}")

# Find the node /encoder/0/encoder_pos/If
target_node_name = "/encoder/0/encoder_pos/If"
target_node = None

for node in model.graph.node:
    if node.name == target_node_name:
        target_node = node
        break

if target_node:
    print(f"Found node: {target_node.name} ({target_node.op_type})")
    print("Inputs:", target_node.input)
    print("Outputs:", target_node.output)
    
    # Inspect attributes (subgraphs)
    for attr in target_node.attribute:
        print(f"Attribute: {attr.name}")
        if attr.type == onnx.AttributeProto.GRAPH:
            subgraph = attr.g
            print(f"  Subgraph: {subgraph.name}")
            print("  Subgraph Inputs:", [i.name for i in subgraph.input])
            print("  Subgraph Outputs:", [o.name for o in subgraph.output])
            
            # Check the type of the output in the subgraph
            # We need to look at the last node in subgraph producing the output, or ValueInfo?
            # Subgraph outputs are defined in subgraph.output
            for out in subgraph.output:
                print(f"    Output: {out.name}, Type: {out.type}")
else:
    # If name not found, search by op_type and heuristics
    print(f"Node {target_node_name} not found by exact name match. Searching for 'If' nodes...")
    for i, node in enumerate(model.graph.node):
        if node.op_type == "If":
            print(f"Possible candidate [{i}]: {node.name}")
            # Just print the first one for now
            target_node = node
            break
            
    if target_node:
        print(f"Inspecting first If node: {target_node.name}")
        for attr in target_node.attribute:
             if attr.type == onnx.AttributeProto.GRAPH:
                subgraph = attr.g
                print(f"  Subgraph Outputs ({attr.name}):")
                for out in subgraph.output:
                    print(f"    Name: {out.name}")
                    if out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
                        print("    Type: FLOAT (1)")
                    elif out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
                        print("    Type: FLOAT16 (10)")
                    else:
                        print(f"    Type: {out.type.tensor_type.elem_type}")

# Also find Cast_3
target_cast_name = "/encoder/0/encoder_pos/Cast_3"
print(f"\nSearching for {target_cast_name}...")

found_cast = False
for node in model.graph.node:
    if node.name == target_cast_name:
        print(f"Found node: {node.name} ({node.op_type})")
        print("Inputs:", node.input)
        print("Outputs:", node.output)
        found_cast = True
        break

if not found_cast:
    print("Cast_3 not found in main graph. Searching in subgraphs...")
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                subgraph = attr.g
                for sub_node in subgraph.node:
                     if sub_node.name == target_cast_name:
                        print(f"Found node inside {node.name}/{attr.name}: {sub_node.name} ({sub_node.op_type})")
                        print("Inputs:", sub_node.input)
                        print("Outputs:", sub_node.output)
                        # Check attributes if it's a Cast node to see 'to' type
                        for sattr in sub_node.attribute:
                            print(f"  Attribute: {sattr.name} = {sattr}")
                        found_cast = True
                        
                        # Check ValueInfo for the output
                        output_name = sub_node.output[0]
                        print(f"Checking ValueInfo for {output_name}...")
                        found_vi = False
                        # ValueInfo is in graph.value_info or graph.output (if it's a graph output)
                        # But this is a subgraph. It might return the value_info locally if defined?
                        # Or checking the main graph value_info?
                        # Usually convert_float_to_float16 repopulates value_info.
                        
                        # Check subgraph value_info
                        for vi in subgraph.value_info:
                            if vi.name == output_name:
                                print(f"  Found in subgraph value_info: {vi.name}")
                                print(f"  Type: {vi.type.tensor_type.elem_type}")
                                found_vi = True
                        
                        # Check main graph value_info
                        for vi in model.graph.value_info:
                             if vi.name == output_name:
                                print(f"  Found in main graph value_info: {vi.name}")
                                print(f"  Type: {vi.type.tensor_type.elem_type}")
                                found_vi = True
                                
                        if not found_vi:
                            print("  ValueInfo not found explicitly.")
                            
                        break
        if found_cast: break
