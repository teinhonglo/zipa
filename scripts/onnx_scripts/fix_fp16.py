
import onnx
import sys
import os

def fix_model(model_path):
    print(f"Fixing {model_path}...")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return False

    fixed_count = 0
    
    # Helper to process graph and subgraphs recursively
    def process_graph(graph):
        nonlocal fixed_count
        # Build map of value info types for checking
        val_info_types = {}
        for vi in graph.value_info:
             val_info_types[vi.name] = vi.type.tensor_type.elem_type
        for vi in graph.output:
             val_info_types[vi.name] = vi.type.tensor_type.elem_type
        for vi in graph.input:
             val_info_types[vi.name] = vi.type.tensor_type.elem_type
             
        # Iterate over nodes
        for node in graph.node:
            # Process subgraphs first
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    process_graph(attr.g)
            
            # Check Cast nodes
            if node.op_type == "Cast":
                # Check 'to' attribute
                to_attr = None
                for attr in node.attribute:
                    if attr.name == "to":
                        to_attr = attr
                        break
                
                if to_attr and to_attr.i == onnx.TensorProto.FLOAT:
                    # Check if output is actually FLOAT16
                    out_name = node.output[0]
                    
                    found_type = None
                    if out_name in val_info_types:
                        found_type = val_info_types[out_name]
                    else:
                        # Fallback to checking main graph value info if not found in subgraph
                        # (simplification: assume we can access main graph value infos if strictly needed, 
                        # but usually subgraph value infos are self-contained or passed in)
                        # For now rely on local graph info.
                        pass
                    
                    if found_type == onnx.TensorProto.FLOAT16:
                        print(f"  Fixing Cast node {node.name} (Output: {out_name}): FLOAT -> FLOAT16")
                        to_attr.i = onnx.TensorProto.FLOAT16
                        fixed_count += 1

    process_graph(model.graph)
    
    if fixed_count > 0:
        print(f"  Fixed {fixed_count} nodes. Saving...")
        onnx.save(model, model_path)
        return True
    else:
        print("  No nodes needed fixing.")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # explicit path
        paths = sys.argv[1:]
    else:
        # Auto-discover all .fp16.onnx files in checkpoints
        import glob
        paths = glob.glob("checkpoints/*/*/*.fp16.onnx")
    
    for p in paths:
        fix_model(p)
