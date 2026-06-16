import json
import glob
import pathlib

import polars


def extract_features(path: pathlib.Path) -> dict:
    data = json.load(open(path))

    text = data["full_text"]
    entities = data.get("entities", {})

    return {
        "idx": path.stem,
        "text": text,
        **entities
    }


files = [
    extract_features(pathlib.Path(file))
    for file in glob.glob("data/raw/**/*.json")
]

dataframe = (
    polars.from_dicts(files)
    .drop(["reference_numbers", "dates"])
)

print(
    dataframe["people"]
    .explode()
    .drop_nulls()
    .str.to_lowercase()
    # name renames
    .str.replace(r"^epstein$", "jeffrey epstein")
    .str.replace("ms. maxwell", "ghislaine maxwell")
    # .str.replace(r"^maxwell$", "ghislaine maxwell")
    # ---
    .value_counts(sort=True)
    .write_csv("data/interim/count.people.csv")
)

print(
    dataframe["organizations"]
    .explode()
    .drop_nulls()
    .str.to_lowercase()
    #
    .str.replace(r"^doj$", "department of justice")
    #
    .value_counts(sort=True)
    .write_csv("data/interim/count.organizations.csv")
)

print(
    dataframe["locations"]
    .explode()
    .drop_nulls()
    .str.to_lowercase()
    .value_counts(sort=True)
    .write_csv("data/interim/count.locations.csv")
)