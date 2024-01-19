import os
import pandas as pd

from converters.ans_to_int import AnsToNumConverter
from exceptions import UnknownAnsiSequence, KnownAnsiSequenceUnimplemented
from data_selectors.obsolete_training_selector import ObsoleteTrainingSelector
from translators.translation_utils import CharToVecTranslator

selector1 = ObsoleteTrainingSelector(inner_radius=1, outer_radius=0, ignore_downstream=True)
selector2 = ObsoleteTrainingSelector(inner_radius=3, outer_radius=3, ignore_downstream=True)
converter = AnsToNumConverter(translator=CharToVecTranslator)
samples1 = []
samples2 = []


for year in range(1990, 2024):
    samples1 = []
    samples2 = []

    print(f"\n{year}: ", end="")
    for index, file in enumerate(os.listdir(f"ansi_files/{year}/")):
        if index % 20 == 0:
            print(".", end="")
        try:
            filepath = f"ansi_files/{year}/{file}"
            lines = converter.extract_ansi_lines(filepath)
            x = converter.convert(lines, filepath=filepath)
            samples1.extend(selector1.get_training_examples(x))
            samples2.extend(selector2.get_training_examples(x))
            print(end="")
        except NotImplementedError as ex:
            print(ex)
        except UnknownAnsiSequence as ex:
            print(ex)
        except KnownAnsiSequenceUnimplemented as ex:
            print(ex)
        except (IndexError, ValueError) as ex:
            print(ex)
        except Exception as ex:
            print(f"ERROR ansi_files/{year}/{file} {ex}")

    pd.DataFrame(samples1).to_csv(f"samples1_{year}.csv")
    pd.DataFrame(samples2).to_csv(f"samples2_{year}.csv")


print(converter.unknown_ansi_sequences)


#y = Ans2IntConverter().convert("ansi_files/sample_chars.ans")
