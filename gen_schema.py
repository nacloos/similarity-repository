"""
Preliminary attempt to automatically generate cue schemas from python modules.
"""
from config_utils.gen_schema import generate_module_schemas


generate_module_schemas(
    "rsatoolbox/rdm",
    pkg_name="rdm",
    jsonschema_dir=".jsonschema",
    cue_dir="similarity/cue.mod/pkg/github.com"
)

generate_module_schemas(
    "netrep/metrics",
    pkg_name="metrics",
    jsonschema_dir=".jsonschema",
    cue_dir="similarity/cue.mod/pkg/github.com"
)
