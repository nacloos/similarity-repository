import os
from mouse_vision.core.default_dirs import NEURAL_DATA_DIR

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_project_api import EcephysProjectWarehouseApi
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine

manifest_path = os.path.join(NEURAL_DATA_DIR, 'manifest.json')
cache = EcephysProjectCache(manifest=manifest_path,
                            fetch_api=EcephysProjectWarehouseApi(RmaEngine(
                                scheme="http",
                                host="api.brain-map.org",
                                timeout=50 * 60 # set timeout to 50 minutes
                                ))
                            )
# here we now extract only brain observatory data (as opposed to functional connectivity)
sessions = cache.get_session_table()
brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
session_ids = brain_observatory_type_sessions.index.values

# actually download the data
for session_id in session_ids:
    print('Downloading session id: {}'.format(session_id))
    session = cache.get_session_data(session_id)
