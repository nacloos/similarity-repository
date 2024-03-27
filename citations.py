"""
Automatically generate citation list from measures and backends.
"""
import numpy as np
import similarity


def get_last_name(citation):
    # this is not perfect (e.g. doesn't work if there is only one author)
    authors = citation.split(", ")
    if len(authors) > 0:
        last_name = authors[0].split()[-1]
        return last_name
    return ""


def generate_citations():
    def _add_papers(paper_ids, citations):
        if isinstance(paper_ids, str):
            paper_ids = [paper_ids]

        for paper_id in paper_ids:
            paper = similarity.make(f"paper.{paper_id}")
            if "citation" in paper:
                citations.add(paper["citation"])
            else:
                print(f"Paper {paper_id} does not have a citation. Skipping.")

    citations = set()
    # go through all cards
    cards = similarity.make("card.*")
    for k, card in cards.items():
        if "paper" in card:
            _add_papers(card["paper"], citations)

    ids = similarity.match("measure.*")
    backends = np.unique([id.split(".")[1] for id in ids])
    for backend in backends:
        backend_card = similarity.make(f"measure.{backend}")
        if "paper" in backend_card:
            _add_papers(backend_card["paper"], citations)

    citations = sorted(citations, key=get_last_name)
    return citations


if __name__ == "__main__":
    citations = generate_citations()
    print(f"Generated {len(citations)} citations")
    text = ""
    for citation in citations:
        text += f"{citation}\n\n"

    with open("citations.txt", "w", encoding="utf-8") as f:
        f.write(text)
