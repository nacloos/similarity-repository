"""
Automatically generate citation list from measures and backends.
"""
import similarity


def get_last_name(citation):
    # TODO: this is not perfect (e.g. doesn't work if there is only one author)
    authors = citation.split(", ")
    if len(authors) > 0:
        last_name = authors[0].split()[-1]
        return last_name
    return ""


def generate_citations():

    citations = set()

    measure_configs = similarity.make("measure", return_config=True)
    for k, cfg in measure_configs.items():
        if "paper" in cfg:
            if isinstance(cfg["paper"], list):
                for paper in cfg["paper"]:
                    citations.add(paper["citation"])
            else:
                citations.add(cfg["paper"]["citation"])

    backends = similarity.make("backend", return_config=True)
    for k, backend in backends.items():
        if "paper" in backend:
            if isinstance(backend["paper"], list):
                for paper in backend["paper"]:
                    citations.add(paper["citation"])
            else:
                citations.add(backend["paper"]["citation"])

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
