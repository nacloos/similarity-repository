// package similarity


// // TODO: website with all the cards + implementations
// // TODO: typst table from cards

// // TODO?
// // procrustes: {
// //     card: {}
// //     implementation: {}
// //     distance: {}
// //     score: {}
// // }

// #Card: {
//     pretty_name: string
//     invariance: [..._]
//     reference: [..._]

//     // TODO: put that in card?
//     implementation?: _
// }

// card: {
//     [name=string]: #Card

//     procrustes: {
//         pretty_name: "Procrustes"
//         invariance: [
//             #isotropic_scaling, 
//             #orthogonal
//         ]
//         // TOOD: @bibtex("...")
//         reference: [
//             {
//                 title: "Generalized Shape measures on Neural Representations"
//                 author: "Alex H. Williams and Erin Kunz and Simon Kornblith and Scott W. Linderman"
//                 year: "2022"
//                 eprint: "2110.14739"
//                 archivePrefix: "arXiv"
//                 primaryClass: "stat.ML"
//             },
//             {bibtex: """
//                 @misc{williams2022generalized,
//                     title={Generalized Shape measures on Neural Representations}, 
//                     author={Alex H. Williams and Erin Kunz and Simon Kornblith and Scott W. Linderman},
//                     year={2022},
//                     eprint={2110.14739},
//                     archivePrefix={arXiv},
//                     primaryClass={stat.ML}
//                 }
//             """},
//             {
//                 title: "Grounding Representation Similarity Through Statistical Testing"
//                 author: "Frances Ding and Jean-Stanislas Denain and Jacob Steinhardt"
//                 booktitle: "Advances in Neural Information Processing Systems"
//                 editor: "A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan"
//                 year: 2021
//                 url: "https://openreview.net/forum?id=_kwj6V53ZqB"
//             }
//         ]

//         // or backend
//         implementation: {
//             netrep: {


//             }
//         }
//     }

//     cca: {
//         pretty_name: "Canonical Correlation Analysis"
//         invariance: [#invertible_linear]
//         reference: [

//         ]
//     }

//     cka: {
//         pretty_name: "Centered Kernel Alignment"
//         invariance: [#isotropic_scaling, #orthogonal]
//         reference: []
//     }
// }



