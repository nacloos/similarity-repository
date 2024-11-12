import similarity
import msid


similarity.register(
    "measure.imd",
    {
        "paper_id": "tsitsulin2020",
        "github": "https://github.com/xgfs/imd"
    }
)

similarity.register(
    "measure.imd.imd",
    msid.msid_score,
    function=True,
    preprocessing=[
        "reshape2d",
    ]
)
