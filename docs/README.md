# Pseudocode Design Docs

```python
import daam

model = DiffusionPipeline()
prompts: List[str]

for prompt in prompts:
    with daam.trace(daam.UnetCrossAttentionLocator(model)) as trace:
        model(prompt)
        heat_maps = trace.compute_heat_maps()
    
    for heat_map in heat_maps:
        print(heat_map.word)
        heat_map.image.show()
```