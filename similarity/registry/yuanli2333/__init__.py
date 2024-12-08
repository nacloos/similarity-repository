import similarity
from .CKA import linear_CKA

similarity.register(
    "measure/yuanli2333",
    {
        "github": "https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment"
    }
)

similarity.register(
    "measure/yuanli2333/cka",
    linear_CKA,
)