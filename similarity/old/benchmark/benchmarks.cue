package similarity

#Benchmark: #target & {
    #path: string  // path to the python code
    #reference: string
    #dependencies: [...string]
    ...
}

klabunde23_dimensionality: #Benchmark & {
    #path: "similarity.papers.klabunde23.experiments.benchmark_dimensionality"
    #reference: "https://github.com/mklabunde/survey_measures/blob/8a413cc1bb7de23b41527460999e665a40604604/appendix_procrustes.ipynb"
    #dependencies: ["numpy", "seaborn", "scipy", "pandas", "matplotlib"]
    #partial: true
    
    // dimensionalities = [1, 5, 10, 20, 30, 100, 200, 300]
    noise_levels: [1e-2, 5e-2, 1e-1]
    dimensionalities: [1, 10, 100]
    // noise_levels: [1e-2]
    N: 1000
    D: 10
    n_repetitions: 1
    // n_repetitions: 10
    n_permutations: 10

    // measure: #target
    // save_path: "figures/klabunde23/benchmark.png"
}


klabunde23_procrustes: klabunde23_dimensionality & {
    // measure: #target & {
    //     #path: "similarity.papers.klabunde23.experiments.procrustes"
    //     #partial: true
    // }
    measure: procrustes
}

// benchmark: klabunde23: cka: klabunde23_dimensionality & {
klabunde23_cka: klabunde23_dimensionality & {
    measure: cka & {#partial: true}
}