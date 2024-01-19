import asyncio
import json
import os
import re
import zipfile
from json import JSONDecodeError
from urllib.request import urlopen
import io
import pandas as pd

import logging

import aiohttp as aiohttp

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# those 11 are involved in a side-effect!
art_data_by_filename = {}
metadata_columns_by_name = {
 name: [] for name in ("year", "pack", "filename", "archive", "filesize", "legacy_aspect", "letter_spacing", "artist", "group", "content")
}

def main():
    os.makedirs("ansi_files", exist_ok=True)

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for year in range(1990, 2024):
        print(year)
        asyncio.run(process_year(year))

    #with open("ansi_files/ansi_data_by_filename.json", "w") as file:
    #    json.dump(art_data_by_filename, fp=file, )

    tabular_metadata = pd.DataFrame(metadata_columns_by_name)
    tabular_metadata.to_csv("ansi_files/metadata.csv")

async def process_year(year):
    page = 0
    while True:
        page += 1
        try:
            website = await read_as_html(f"https://16colo.rs/year/{year}/{page}")
        except Exception as ex:
            logger.error(f"Could not process year {year}, page {page}")
            logger.exception(ex)
            return
        pack_pattern = r'<a href="/pack/([^/]+)/" class="dizname block">'
        matches = re.findall(pack_pattern, website)
        for match in set(matches):
            await process_pack(match, year)

        if not matches:
            break


async def process_pack(pack_name, year):
    try:
        pack_page = await read_as_html(f"https://16colo.rs/pack/{pack_name}/")
        art_pattern = f"<a href=\"/pack/{pack_name}/([^/]+)\">"
        art_matches = re.findall(art_pattern, pack_page)
        for art_name in set(art_matches):
            await process_ansi_art(art_name, pack_name, year)
    except Exception as ex:
        logger.error(f"Unable to process pack {pack_name} with problems due to {ex.__class__} {str(ex)}")


async def process_ansi_art(art_name, pack_name, year):
    ans_filename = None
    try:
        os.makedirs(f"ansi_files/{year}", exist_ok=True)

        if dict(enumerate(art_name.split("."))).get(1) == "ANS":
            # art_page = read_page_as_html(f"https://16colo.rs/pack/{match}/{art_match}")
            ans_filename = f"{pack_name}_{art_name}"
            metadata = await get_metadata_for_art(art_name, pack_name)
            art_data_by_filename[ans_filename] = metadata
            process_ansi_metadata_with_sideeffect(metadata)
            #with open(os.path.join(getcwd(), f"ansi_files/{year}/{ans_filename}"), "w") as file:
            #     raw_art = urlopen(f"https://16colo.rs/pack/{pack_name}/raw/{art_name}")
            #     for line in raw_art.fp:
            #         file.write(line.decode("cp1252"))
    except UnicodeDecodeError as ex:
        logger.debug(f"\n{ex.__class__} while working on {ans_filename} from https://16colo.rs/pack/{pack_name}/{art_name} {str(ex)}")
    except Exception as ex:
        logger.error(f"\n{ex.__class__} while working on {ans_filename} from https://16colo.rs/pack/{pack_name}/{art_name} {str(ex)}")


async def get_metadata_for_art(art_name, pack_name):
    try:
        return json.loads(await read_as_html(f"https://16colo.rs/pack/{pack_name}/data/{art_name}"))
    except JSONDecodeError:
        return {}

def process_ansi_metadata_with_sideeffect(metadata) -> None:
    metadata_columns_by_name["year"].append(metadata.get("year"))
    metadata_columns_by_name["pack"].append(metadata.get("pack"))
    metadata_columns_by_name["filename"].append(metadata.get("filename"))
    metadata_columns_by_name["archive"].append(metadata.get("archive"))
    metadata_columns_by_name["filesize"].append(metadata.get("filesize"))
    metadata_columns_by_name["legacy_aspect"].append(metadata.get("legacy-aspect"))
    metadata_columns_by_name["letter_spacing"].append(metadata.get("letter-spacing"))
    metadata_columns_by_name["artist"].append(", ".join(metadata.get("tags", {}).get("artist", [])))
    metadata_columns_by_name["group"].append(", ".join(metadata.get("tags", {}).get("group", [])))
    metadata_columns_by_name["content"].append(", ".join(metadata.get("tags", {}).get("content", [])))

async def read_as_html(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


if __name__ == "__main__":
    main()


async def read_zip_file_from_url(url: str) -> None:
    payload: bytes = urlopen(url).read()
    zip_data = io.BytesIO(payload)
    async with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        zip_ref.extractall("./ansi_files")