// package test
// import(
//     "strings"
//     "github.com/similarity/metric:card"
//     "github.com/similarity/utils"
// )


// // error if use same name for cards and cards with variants
// _cards: card.cards
// cards: {
//     for key, card in _cards if (card.parameters != _|_ && key == "permutation") {
//         for p in (utils.#Cartesian & {inp: card.parameters}).out {
//             let metric_name = key + "-" + strings.Join(p, "-")
//             (metric_name): {
//                 "name": card.name
//             }
//         }
//     }
//     ...
// }
