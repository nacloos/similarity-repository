from pathlib import Path
import similarity


# path to build directory
BUILD_DIR = Path(__file__).parent / "similarity" / "types"


def build(build_dir=BUILD_DIR):
    """
    Save registered keys as a literal type. Used to get autocomplete for the make function.
    """
    print("Building id types...")
    similarity.registration._register_imports()

    ids = similarity.registration.registry.keys()
    ids = sorted(ids)

    # save literal type for all possible ids
    code = "# Automatically generated code. Do not modify.\n"
    code += "from typing import Literal\n"
    code += "\n\n"
    literal_type = ",\n\t".join([f'"{p}"' for p in ids])
    literal_type = "Literal[\n\t" + literal_type + "\n]"

    code += f"IdType = {literal_type}\n"

    # save type for measure ids only
    code += "\n\n"
    measure_ids = similarity.match("measure.*")
    measure_ids = sorted(measure_ids)
    # remove 'measure.' prefix
    measure_ids = [m.split("measure.", 1)[1] for m in measure_ids]
    literal_type = ",\n\t".join([f'"{p}"' for p in measure_ids])
    literal_type = "Literal[\n\t" + literal_type + "\n]"
    code += f"MeasureIdType = {literal_type}\n"

    with open(build_dir / "__init__.py", "w") as f:
        f.write(code)


if __name__ == "__main__":
    from time import perf_counter
    tic = perf_counter()
    build()
    print(f"Built in {perf_counter() - tic:.2f}s")
