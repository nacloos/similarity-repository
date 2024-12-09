import similarity
import msid


similarity.register("imd/imd", msid.msid_score)