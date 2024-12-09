import similarity


papers = {
    "repsim": {
        "id": "lange2023",
        "shortcite": "Lange et al., 2023",
        "github": "https://github.com/wrongu/repsim/"
    },
    "rsatoolbox": {
        "id": "kriegeskorte2008",
        "shortcite": "Kriegeskorte et al., 2008",
        "github": "https://github.com/rsagroup/rsatoolbox"
    },
    "rtd": {
        "id": "barannikov2022",
        "shortcite": "Barannikov et al., 2022",
        "github": "https://github.com/IlyaTrofimov/RTD",
        "paper": "https://arxiv.org/pdf/2201.00058"
    },
    "fsd": {
        "id": "jung2023",
        "shortcite": "Jung et al., 2023",
        "github": "https://github.com/maroo-sky/FSD"
    },
    "pyrcca": {
        "id": "bilenko2016",
        "shortcite": "Bilenko et al., 2016",
        "github": "https://github.com/gallantlab/pyrcca"
    },
    "xrsa_awes": {
        "id": "ravishankar2021",
        "shortcite": "Ravishankar et al., 2021",
        "github": "https://github.com/uds-lsv/xRSA-AWEs",
        "paper": "https://arxiv.org/pdf/2109.10179"
    },
    "subspacematch": {
        "id": "wang2018",
        "shortcite": "Wang et al., 2018",
        "github": "https://github.com/MeckyWu/subspace-match"
    },
    "representation_similarity": {
        "id": "kornblith2019",
        "shortcite": "Kornblith et al., 2019",
        "github": "https://github.com/google-research/google-research/tree/master/representation_similarity",
        "paper": "https://arxiv.org/abs/1806.05759"
    },
    "netrep": {
        "id": "williams2021",
        "shortcite": "Williams et al., 2021",
        "github": "https://github.com/ahwillia/netrep"
    },
    "brainscore": {
        "id": "schrimpf2018",
        "shortcite": "Schrimpf et al., 2018",
        "github": "https://github.com/brain-score/brain-score"
    },
    "deepdive": {
        "id": "conwell2023",
        "shortcite": "Conwell et al., 2023",
        "github": "https://github.com/ColinConwell/DeepDive"
    },
    "survey_measures": {
        "id": "klabunde2023",
        "shortcite": "Klabunde et al., 2023",
        "github": "https://github.com/mklabunde/survey_measures"
    },
    "nn_similarity_index": {
        "id": "tang2020",
        "shortcite": "Tang et al., 2020",
        "github": "https://github.com/amzn/xfer/blob/master/nn_similarity_index"
    },
    "platonic": {
        "id": "huh2024prh",
        "shortcite": "Huh et al., 2024",
        "github": "https://github.com/minyoungg/platonic-rep"
    },
    "sim_metric": {
        "id": "ding2021",
        "shortcite": "Ding et al., 2021",
        "github": "https://github.com/js-d/sim_metric"
    },
    "subspacematch": {
        "id": "wang2018",
        "shortcite": "Wang et al., 2018",
        "github": "https://github.com/MeckyWu/subspace-match"
    },
    "svcca": {
        "id": "raghu2017",
        "shortcite": "Raghu et al., 2017",
        "github": "https://github.com/google/svcca",
        "paper": "https://arxiv.org/abs/1706.05806"
    },
    "neuroaimetrics": {
        "id": "soni2024",
        "shortcite": "Soni et al., 2024",
        "github": "https://github.com/anshksoni/NeuroAIMetrics"
    },
    "llm_repsim": {
        "id": "klabunde2023",
        "shortcite": "Klabunde et al., 2023",
        "github": "https://github.com/mklabunde/llm_repsim"
    },
    "imd": {
        "id": "tsitsulin2020",
        "shortcite": "Tsitsulin et al., 2020",
        "github": "https://github.com/xgfs/imd"
    },
    "correcting_cka_alignment": {
        "id": "murphy2024",
        "shortcite": "Murphy et al., 2024",
        "github": "https://github.com/Alxmrphi/correcting_CKA_alignment"
    },
    "implicitdeclaration_similarity": {
        "id": "chen2022",
        "shortcite": "Chen et al., 2022",
        "github": "https://github.com/implicitDeclaration/similarity"
    },
    "stir": {
        "id": "nanda2022",
        "shortcite": "Nanda et al., 2022",
        "github": "https://github.com/nvedant07/STIR"
    },
    "contrasim": {
        "id": "rahamim2024",
        "shortcite": "Rahamim et al., 2024",
        "github": "https://github.com/technion-cs-nlp/ContraSim"
    },
    "resi": {
        "id": "klabunde2024",
        "shortcite": "Klabunde et al., 2024",
        "github": "https://github.com/mklabunde/resi"
    },
    "thingsvision": {
        "id": "muttenthaler2021",
        "shortcite": "Muttenthaler et al., 2021",
        "github": "https://github.com/ViCCo-Group/thingsvision"
    },
    "drfrankenstein": {
        "id": "csiszárik2021",
        "shortcite": "Csiszárik et al., 2021",
        "github": "https://github.com/renyi-ai/drfrankenstein"
    },
    "ensd": {
        "id": "giaffar2023",
        "shortcite": "Giaffar et al., 2023",
        "github": "https://github.com/camillerb/ENSD"
    },
    "nnsrm_neurips18": {
        "id": "lu2018",
        "shortcite": "Lu et al., 2018",
        "github": "https://github.com/qihongl/nnsrm-neurips18"
    },
    "brain_language_nlp": {
        "id": "toneva2019",
        "shortcite": "Toneva et al., 2019",
        "github": "https://github.com/mtoneva/brain_language_nlp"
    },
    "mouse_vision": {
        "id": "nayebi2023",
        "shortcite": "Nayebi et al., 2023",
        "github": "https://github.com/neuroailab/mouse-vision"
    },
    "modelsym": {
        "id": "godfrey2023",
        "shortcite": "Godfrey et al., 2023",
        "github": "https://github.com/pnnl/modelsym"
    },
    "unsupervised_analysis": {
        "id": "gwilliam2022",
        "shortcite": "Gwilliam et al., 2022",
        "github": "https://github.com/mgwillia/unsupervised-analysis"
    }
}


for k, v in papers.items():
    similarity.register(f"paper/{k}", v)