package card
import(
    "math"
    "strings"
    "github.com/similarity/utils"
)


#MetricCard: {
    name: string
    paper?: #Paper | [...#Paper]
    invariance?: [...#GroupTransformation]
    // only string because used to generate metric id (TODO: allow number and convert to string?)
    parameters?: [string]: [...string]
    defaults?: _
    // TODO: do the same with paper? just store the id of the paper?
    // store the id of the poperty so that can easily filter by property
    properties: [...#PropertyId] | *[]
    naming?: string
}

#Paper: {
    citation?: string | [...string]
    github?: string
    bibtex?: string
}

property: {
    "permutation-invariant": {},
    "scale-invariant": {
        name: "Scale Invariant"
    }
    "rotation-invariant": {
        name: "Rotation Invariant"
    }
    "invertible-linear-invariant": {}
    "translation-invariant": {}
    "affine-invariant": {}
    score: {
        name: "Score"
        description: "Measure of similarity. A score of 1 indicates that the representations are identical."
    }
    metric: {
        name: "Metric"
        description: "Measure of distance."
    }
}
#PropertyId: or([for k, v in property { k }])

// TODO: keys in cards define the MetricName type
// #MetricName: or([for k, _ in cards { k }])

// TODO?: categories: ["cca", "alignment", "rsm", "neighbors, ...]
// useful for filering (e.g. backend table only for alignement and cca mertrics)


cards: {
    for key, card in _cards {
        if card.parameters == _|_ { (key): card }
        if card.parameters != _|_ {
            for p in (utils.#Cartesian & {inp: card.parameters}).out {
                let metric_name = key + "-" + strings.Join(p, "-")
                (metric_name): card
            }
            // TODO: default params
            (key): card
        }
    }
}

// describe metrics independently of any implementation
cards: [string]: #MetricCard
_cards: {
    ...
    // TODO: scoring_method?
    permutation: {
        name: "Permutation"
        // TODO: order of the parameters? => user naming
        parameters: {
            score_method: ["euclidean", "angular"]
        }
        // TODO: cue to generate names?
        naming: "score_method"
        // properties: [
        //     "permutation-invariant"
        // ]
    }
    correlation: {
        name: "Correlation"
    }
    // for score_method in ["euclidean", "angular"] {
    //     ("permutation-" + score_method): {
    //         name: "Permutation distance-\(score_method)"
    //     }
    // }
 
    // canonical correlation analysis
    cca: {  // TODO: call it cca or mean_cca?
        // TODO: the survey has two rows for mean cc, ok to merge them here?
        name: "Mean Canonical Correlation"
        paper: [papers.yanai1974, papers.raghu2017, papers.kornblith2019]
        parameters: {
            scoring_method: ["euclidean", "angular"]
        }
        // properties: [
        //     "score",
        //     "permutation-invariant",
        //     "scale-invariant",
        //     "rotation-invariant",
        //     "translation-invariant",
        //     "affine-invariant",
        //     "invertible-linear-invariant",
        // ]
    }
    cca_mean_sq_corr: {
        name: "Mean Squared Canonical Correlation"
        // properties: [
        //     "score"
        // ]
    }

    svcca: {
        name: "Singular Vector Canonical Correlation Analysis"
        paper: papers.raghu2017
        parameters: {
            variance_fraction: ["var95", "var99"]
        }
        // properties: [
        //     "score",
        //     "permutation-invariant",
        //     "scale-invariant",
        //     "rotation-invariant",
        //     "translation-invariant",
        // ]
    }
    pwcca: {
        name: "Projection-Weighted Canonical Correlation Analysis"
        paper: papers.morcos2018
    }

    "riemannian_metric": {
        name: "Riemannian Metric"
        paper: papers.shahbazi2021
    }

    procrustes: {
        name: "Orthogonal Procrustes"
        paper: [papers.ding2021, papers.williams2021]
        parameters: {
            scoring_method: ["euclidean", "angular"]
        }
        // properties: [
        //     "metric",
        //     "permutation-invariant",
        //     "scale-invariant",
        //     "rotation-invariant"
        // ]
    }
    // TODO: use argument? e.g. squared_or_not: ["sq", null]
    "procrustes-sq": procrustes

    "procrustes-score": {
        name: "Procrustes Score"
        // properties: [
        //     "score"
        // ]
    }

    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
        "shape_metric-angular-alpha\(math.Round(alpha*10))e-1": {
            name: "Angular Shape Metric"
            paper: papers.williams2021
            // TODO: alpha=0 equivalant to cca, alpha=1 to procrustes
            // parameters: {
            //     alpha: ["alpha0", "alpha0.5", "alpha1"]
            // }
            // properties: ["metric"]
        }
        "shape_metric-euclidean-alpha\(math.Round(alpha*10))e-1": {
            name: "Euclidean Shape Metric"
            paper: papers.williams2021
            // properties: ["metric"]
        }
    }
    partial_whitening_shape_metric: {
        name: "Partial-Whitening Shape Metric"
        paper: papers.williams2021
    }
    // TODO
    pls: {
        name: "Partial Least Squares"
        // paper: TODO
    }
    linear_regression: {
        name: "Linear Regression"
        paper: [papers.li2016, papers.kornblith2019]
    }
    aligned_cosine: {
        name: "Aligned Cosine Similarity"
        paper: papers.hamilton2016a
    }
    corr_match: {
        name: "Correlation Match"
        paper: papers.li2016
    }
    max_match: {
        name: "Maximum Match"
        paper: papers.wang2018
    }

    // representational similarity matrix
    rsm_norm: {
        name: "Representational Similarity Matrix Norms"
        paper: [papers.shahbazi2021, papers.yin2018]
    }

    rsa: {
        name: "Representational Similarity Analysis"
        paper: papers.kriegeskorte2008
        // properties: [
        //     "score",
        //     // TODO: may depend on the specific implementation and preprocessing
        //     // make it possible to overwrite the default properties?
        //     "permutation-invariant",
        //     "scale-invariant",
        //     "translation-invariant"
        // ]
        parameters: {
            rdm_method: [
                "euclidean",
                "correlation",
                "mahalanobis",
                "crossnobis",
                "poisson",
                "poisson_cv"
            ]
            compare_method: [
                "cosine",
                "spearman",
                "corr",
                "kendall",
                "tau_b",
                "tau_a",
                "rho_a",
                "corr_cov",
                "cosine_cov",
                "neg_riem_dist"
            ]
        }
        // TODO: provide defaults for "rsa"
        defaults: {
            rdm_method: "euclidean"
            compare_method: "cosine"
        }
        // naming: "rsa-{rdm_method}-{compare_method}"
    }    

    cka: {
        name: "Centered Kernel Alignment"
        paper: papers.kornblith2019
        invariance: []
        // properties: [
        //     "score",
        //     "scale-invariant",
        //     "rotation-invariant",
        //     "permutation-invariant",
        //     "translation-invariant"
        // ]
    }
    "cka-angular": {
        name: "Angular CKA"
        paper: [papers.williams2021, papers.lange2022]
        // properties: [
        //     "metric",
        //     // "riemannian-metric",
        //     "scale-invariant",
        //     "rotation-invariant"
        // ]
    }
    dcor: {
        name: "Distance Correlation"
        paper: papers.szekely2007
    }
    nbs: {
        name: "Normalized Bures Similarity"
        paper: papers.tang2020
    }
    eos: {
        name: "Eigenspace Overlap Score"
        paper: papers.may2019
    }
    gulp: {
        name: "Unified Linear Probing"
        paper: papers.boixadsera2022
    }
    "riemmanian_metric": {
        name: "Riemmanian Distance"
        paper: papers.shahbazi2021
        // properties: [
        //     "metric"
        // ]
    }

    // neighbors
    jaccard: {
        name: "Jaccard"
        paper: [papers.schumacher2021, papers.wang2020, papers.hryniowski2020, papers.gwilliam2022]
    }
    second_order_cosine: {
        name: "Second-Prder Cosine Similarity"
        paper: papers.hamilton2016b
    }
    rank_similarity: {
        name: "Rank Similarity"
        paper: papers.wang2020
    }
    joint_rank_jaccard: {
        name: "Joint Rang and Jaccard Similarity"
        paper: papers.wang2020
    }

    // topology
    gs: {
        name: "Geometry Score"
        paper: papers.khrulkov2018
    }
    imd: {
        name: "Multi-scale Intrinsic Distance"
        paper: papers.tsitsulin2020
    }
    rtd: {
        name: "Representation Topology Divergence"
        paper: papers.barannikov2022
    }

    // statistic
    intrinsic_dimension: {
        name: "Intrinsic Dimension"
        paper: papers.camastra2016
    }
    magnitude: {
        name: "Magnitude"
        paper: papers.wang2020
    }
    concentricity: {
        name: "Concentricity"
        paper: papers.wang2020
    }
    uniformity: {
        name: "Uniformity"
        paper: papers.wang2022
    }
    tolerance: {
        name: "Tolerance"
        paper: papers.wang2021
    }
    knn_graph_modularity: {
        name: "kNN-Graph Modularity"
        paper: papers.lu2022
    }
    neuron_graph_modularity: {
        name: "Neuron-Graph Modularity"
        paper: papers.lange2022
    }
}

papers: [string]: #Paper
papers: {
    barannikov2022: {
        citation: "Serguei Barannikov, Ilya Trofimov, Nikita Balabin, and Evgeny Burnaev. 2022. Representation Topology Divergence: A Method for Comparing Neural Network Representations. In ICML."
        github: "https://github.com/IlyaTrofimov/RTD"
    }
    boixadsera2022: {
        citation: "Enric Boix-Adsera, Hannah Lawrence, George Stepaniants, and Philippe Rigollet. 2022. GULP: a prediction-based metric between representations. In NeurIPS."
        github: "https://github.com/sgstepaniants/GULP"
    }
    gwilliam2022: {
        citation: "Matthew Gwilliam and Abhinav Shrivastava. 2022. Beyond Supervised vs. Unsupervised: Representative Benchmarking and Analysis of Image Representation Learning. In CVPR."
        github: "https://github.com/mgwillia/unsupervised-analysis"
    }
    lange2022: {
        citation: "Richard D. Lange, David S. Rolnick, and Konrad P. Kording. 2022. Clustering units in neural networks: upstream vs downstream information. TMLR (2022)."
        github: "https://github.com/wrongu/modularity"
    }
    lu2022: {
        citation: "Yao Lu, Wen Yang, Yunzhe Zhang, Zuohui Chen, Jinyin Chen, Qi Xuan, Zhen Wang, and Xiaoniu Yang. 2022."
        github: "https://github.com/yaolu-zjut/Dynamic-Graphs-Construction"
    }
    wang2022: {
        citation: "Guangcong Wang, Guangrun Wang, Wenqi Liang, and Jianhuang Lai. 2022. Understanding Weight Similarity of Neural Networks via Chain Normalization Rule and Hypothesis-Training-Testing. ArXiv preprint (2022)."
    }
    ding2021: {
        citation: "Frances Ding, Jean-Stanislas Denain, and Jacob Steinhardt. 2021. Grounding Representation Similarity Through Statistical Testing. In NeurIPS."
        github: "https://github.com/js-d/sim_metric"
    }
    shahbazi2021: {
        citation: "Mahdiyar Shahbazi, Ali Shirali, Hamid Aghajan, and Hamed Nili. 2021. Using distance on the Riemannian manifold to compare representations in brain and in models. NeuroImage 239 (2021)."
        github: "https://github.com/mshahbazi1997/riemRSA"
    }
    schumacher2021: {
        citation: "Tobias Schumacher, Hinrikus Wolf, Martin Ritzert, Florian Lemmerich, Martin Grohe, and Markus Strohmaier. 2021. The Effects of Randomness on the Stability of Node Embeddings. In Machine Learning and Principles and Practice of Knowledge Discovery in Databases."
        github: "https://github.com/SGDE2020/embedding_stability"
    }
    williams2021: {
        citation: "Alex H. Williams, Erin Kunz, Simon Kornblith, and Scott W. Linderman. 2021. Generalized Shape Metrics on Neural Representations. In NeurIPS."
        github: "https://github.com/ahwillia/netrep"
    }
    wang2021: {
        citation: "Feng Wang and Huaping Liu. 2021. Understanding the Behaviour of Contrastive Loss. In CVPR."
    }
    hryniowski2020: {
        citation: "Andrew Hryniowski and Alexander Wong. 2020. Inter-layer Information Similarity Assessment of Deep Neural Networks Via Topological Similarity and Persistence Analysis of Data Neighbour Dynamics. ArXiv preprint (2020)."
    }
    tang2020: {
        citation: "Shuai Tang, Wesley J. Maddox, Charlie Dickens, Tom Diethe, and Andreas Damianou. 2020. Similarity of Neural Networks with Gradients. ArXiv preprint (2020)."
        github: "https://github.com/amzn/xfer/tree/master/nn_similarity_index"
    }
    tsitsulin2020: {
        citation: "Anton Tsitsulin, Marina Munkhoeva, Davide Mottin, Panagiotis Karras, Alex Bronstein, Ivan Oseledets, and Emmanuel Mueller. 2020. The Shape of Data: Intrinsic Distance for Data Distributions. In ICLR."
        github: "https://github.com/xgfs/imd"
    }
    wang2020: {
        citation: "Chenxu Wang, Wei Rao, Wenna Guo, Pinghui Wang, Jun Liu, and Xiaohong Guan. 2020. Towards Understanding the Instability of Network Embedding. IEEE TKDE 34, 2 (2020)."
    }
    kornblith2019: {
        citation: "Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey E. Hinton. 2019. Similarity of Neural Network Representations Revisited. In ICML."
    }
    may2019: {
        citation: "Avner May, Jian Zhang, Tri Dao, and Christopher Ré. 2019. On the Downstream Performance of Compressed Word Embeddings. In NeurIPS."
    }
    morcos2018: {
        citation: "Ari S. Morcos, Maithra Raghu, and Samy Bengio. 2018. Insights on representational similarity in neural networks with canonical correlation. In NeurIPS."
        github: "https://github.com/google/svcca"
        bibtex: """
@incollection{NIPS2018_7815,
    title = {Insights on representational similarity in neural networks with canonical correlation},
    author = {Morcos, Ari and Raghu, Maithra and Bengio, Samy},
    booktitle = {Advances in Neural Information Processing Systems 31},
    editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
    pages = {5732--5741},
    year = {2018},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7815-insights-on-representational-similarity-in-neural-networks-with-canonical-correlation.pdf}
}
"""
    }
    khrulkov2018: {
        citation: "Valentin Khrulkov and Ivan Oseledets. 2018. Geometry Score: A Method For Comparing Generative Adversarial Networks. In ICML."
        github: "https://github.com/KhrulkovV/geometry-score"
    }
    wang2018: {
        citation: "Liwei Wang, Lunjia Hu, Jiayuan Gu, Zhiqiang Hu, Yue Wu, Kun He, and John E. Hopcroft. 2018. Towards Understanding Learning Representations: To What Extent Do Different Neural Networks Learn the Same Representation. In NeurIPS."
        github: "https://github.com/MeckyWu/subspace-match"
    }
    yin2018: {
        citation: "Zi Yin and Yuanyuan Shen. 2018. On the Dimensionality of Word Embedding. In NeurIPS."
        github: "https://github.com/ziyin-dl/word-embedding-dimensionality-selection"
    }
    raghu2017: {
        citation: "Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. 2017. SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. In NeurIPS."
        bibtex: """
@incollection{NIPS2017_7188,
    title = {SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability},
    author = {Raghu, Maithra and Gilmer, Justin and Yosinski, Jason and Sohl-Dickstein, Jascha},
    booktitle = {Advances in Neural Information Processing Systems 30},
    editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
    pages = {6076--6085},
    year = {2017},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability.pdf}
}
"""
    }
    camastra2016: {
        citation: "Francesco Camastra and Antonino Staiano. 2016. Intrinsic dimension estimation: Advances and open problems. Information Sciences 328 (2016)."
    }
    hamilton2016b: {
        citation: "William L. Hamilton, Jure Leskovec, and Dan Jurafsky. 2016. Cultural Shift or Linguistic Drift? Comparing Two Computational Measures of Semantic Change. In EMNLP."
    }
    hamilton2016a: {
        citation: "William L. Hamilton, Jure Leskovec, and Dan Jurafsky. 2016. Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change. In ACL."
        github: "https://github.com/williamleif/histwords"
    }
    li2016: {
        citation: "Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John E. Hopcroft. 2016. Convergent Learning: Do different neural networks learn the same representations?. In ICLR."
        github: "https://github.com/yixuanli/convergent_learning"
    }
    kriegeskorte2008: {
        citation: "Nikolaus Kriegeskorte, Marieke Mur, and Peter Bandettini. 2008. Representational similarity analysis - connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience 2 (2008)."
        github: "https://github.com/rsagroup/rsatoolbox"
    }
    szekely2007: {
        citation: "Gábor J. Székely, Maria L. Rizzo, and Nail K. Bakirov. 2007. Measuring and testing dependence by correlation of distances. The Annals of Statistics 35, 6 (2007)."
    }
    yanai1974: {
        citation: "Haruo Yanai. 1974. Unification of Various Techniques of Multivariate Analysis by Means of Generalized Coefficient of Determination. Kodo Keiryogaku (The Japanese Journal of Behaviormetrics) 1 (1974)."
    }
}
