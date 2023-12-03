// package similarity
// import(
//     netrep_measures "github.com/netrep/measures"
// )

// // TODO: measures subpackage or just one netrep package?
// test: netrep_measures.#Linearmeasure & {
//     alpha: 0
// }

// procrustes: #Measure & {
//     #path: "netrep.measures.Linearmeasure"
//     #preprocessing: [#reshape2d]
//     #postprocessing: [#angular_dist_to_score]
//     alpha: 1
// }

// cca: #Measure & {
//     #path: "netrep.measures.Linearmeasure"
//     #preprocessing: [#reshape2d]
//     #postprocessing: [#angular_dist_to_score]
//     alpha: 0
// }

// svcca: #Measure & {
//     #path: "netrep.measures.Linearmeasure"
//     #preprocessing: [#reshape2d, #pca & {n_components: 0.95}]
//     #postprocessing: [#angular_dist_to_score]
//     alpha: 0
// }


// cka: #Measure & {
//     // TODO: directly refer to github file instead of having to copy it here
//     #path: "similarity.measures.yuanli2333.cka.linear_CKA"
//     // don't use call_key because linear_CKA is already a function (not a class with a fit_score method)
//     #call_key: null
//     #partial: true
//     #preprocessing: [#reshape2d]
//     #postprocessing: [#arccos, #angular_dist_to_score]
// }

// #base_rsa: #Measure & {
//     #path: "similarity.measures.rsatoolbox.rsa.compute_rsa"
//     #call_key: null
//     #partial: true
//     #preprocessing: [#reshape2d]
//     #postprocessing: [#arccos, #angular_dist_to_score]
//     rdm_method: string
//     compare_method: string
// }

// rsa: #base_rsa & {
//     rdm_method: "euclidean"
//     compare_method: "cosine"
// }

// // TODO
// // dsa: #Measure & {
// //     #path: "netrep.measures.Linearmeasure"
// //     #preprocessing: [#reshape2d]
// //     #postprocessing: [#angular_dist_to_score]
// //     alpha: 0
// // }