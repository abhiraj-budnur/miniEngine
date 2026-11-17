
import numpy as np
import json
from pathlib import Path

source = Path("tests/reference_outputs.npz")
output_dir = Path("tests/reference")

output_dir.mkdir(parents=True, exist_ok=True)

with np.load(source) as data:

    metadata = {}

    for name in data.files:

        array = data[name]

        if name == "input_ids":

            array = array.astype(np.int32)

            output_file = output_dir / "input_ids.bin"
            array.tofile(output_file)

            metadata[name] = {
                "file": output_file.name,
                "shape": list(array.shape),
                "dtype": "int32",
                "elements": int(array.size)
            }

        elif array.dtype == np.float32:

            output_file = output_dir / f"{name}.bin"

            array.tofile(output_file)

            metadata[name] = {
                "file": output_file.name,
                "shape": list(array.shape),
                "dtype": "float32",
                "elements": int(array.size)
            }

        else:
            metadata[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype)
            }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(metadata, f, indent=4)

print("Reference conversion complete.")

for name, info in metadata.items():
    print(name, info)
