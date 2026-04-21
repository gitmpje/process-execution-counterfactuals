# %%
import torch

model = torch.load("data/model-pe-HingePack-hetero.pth", weights_only=False)
# model = torch.load("data/model-pe-HingePack.pth", weights_only=False)

# %%
print("Model structure:")
print(model)

print("\nDetails of HanConv layers:")
for name, module in model.named_modules():
    print(f"Layer: {name}")
    print(f"  in_channels: {module.in_channels}")
    print(f"  out_channels: {module.out_channels}")
    # Print additional attributes if needed
    if hasattr(module, "lin_rel"):
        print(f"  lin_rel weight shape: {module.lin_rel.weight.shape}")
    if hasattr(module, "lin_root"):
        print(f"  lin_root weight shape: {module.lin_root.weight.shape}")

# %%
dataset = torch.load("data/dataset-pe-HingePack.pt", weights_only=False)
model.eval()

data = dataset[75]
batch_dict = {
    node_type: torch.zeros(
        data[node_type].num_nodes if data[node_type].num_nodes else 0,
        dtype=torch.long,
        device="cpu",
    )
    for node_type in data.node_types
}

out = model(data.x_dict, data.edge_index_dict, batch_dict)
for name, param in model.named_parameters():
    try:
        print(f"{name}: {tuple(param.shape)}")
    except Exception as e:
        print(f"Error printing shape for {name}: {e}")
