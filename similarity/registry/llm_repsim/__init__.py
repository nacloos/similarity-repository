# https://github.com/mklabunde/llm_repsim
from functools import partial
import sys
from pathlib import Path

dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from llmcomp.measures import cka, nearest_neighbor, procrustes, rsa, rsm_norm_difference

import similarity


register = partial(similarity.register, preprocessing=["center_columns"])

register("llm_repsim/cka", cka.centered_kernel_alignment)

register("llm_repsim/jaccard_similarity", nearest_neighbor.jaccard_similarity)

register("llm_repsim/orthogonal_procrustes", procrustes.orthogonal_procrustes)
register("llm_repsim/aligned_cossim", procrustes.aligned_cossim)

register("llm_repsim/representational_similarity_analysis", rsa.representational_similarity_analysis)

register("llm_repsim/rsm_norm_diff", rsm_norm_difference.rsm_norm_diff)
