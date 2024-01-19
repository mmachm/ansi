import shutil

import numpy as np

from converters.ans_from_int import AnsFromNumConverter
from converters.ans_to_int import AnsToNumConverter

converter_to = AnsToNumConverter()
converter_from = AnsFromNumConverter

# lines = converter_to.extract_ansi_lines(f"ansi_files/2007/sadist06_VRG-0306.ANS")
# lines = converter_to.extract_ansi_lines(f"ansi_files/2013/blocktronics_acid_trip_we-ACiDTrip.ANS")
# lines = converter_to.extract_ansi_lines(f"ansi_files/1990/1990_CITYSTAT.ANS")
# lines = converter_to.extract_ansi_lines(f"ansi_files/1990/acdu1190_SUB%231.ANS")
# filename = f"ansi_files/1991/ensiart_A_SMURF.ANS"
filename = f"ansi_files/1991/ensiart_MMSXMAS.ANS"

lines = converter_to.extract_ansi_lines(filename)  # there is a known issue here with some incorrect colors

shutil.copy(filename, "test2.ANS")
# lines = converter_to.extract_ansi_lines(f"ansi_files/sample_chars3.ans")

x = converter_to.convert(lines)
y = AnsFromNumConverter.convert(np.array(x))

with open("test1.ANS", "w", encoding="latin1") as file:
    file.write(y)

lines2 = AnsToNumConverter().extract_ansi_lines(f"test1.ANS")

print()
