import re
from copy import copy
from logging import warning
import tensorflow as tf

import numpy as np

from data_selectors.training_selector import TrainingSelector
from exceptions import UnknownAnsiSequence, KnownAnsiSequenceUnimplemented
from translators.translation_utils import CharToIntTranslator, ModifierTracker

encoding = "latin-1"

unknown_ansi_sequences = []
tf.config.run_functions_eagerly(True)

class DefaultList(list):
    def __init__(self, default_factory):
        super().__init__()
        self._factory = default_factory

    def __getitem__(self, index):
        for _ in range(index - len(self) + 1):
            self.append(self._factory())

        return super().__getitem__(index)

    def __setitem__(self, index, value):
        self.__getitem__(index)
        return super().__setitem__(index, value)


class AnsToNumConverter:
    unknown_ansi_sequences = []

    def get_new_blank_line(self):
        return [copy(self.translator.blank_element) for _ in range(80)]

    def convert_tensor_of_filepaths(self, filepath_tensor):
        results_0, results_1, results_2 = [], [], []
        try:
            for filepath in filepath_tensor:
                #print(filepath)
                try:
                    result = self.convert_filepath(filepath)
                    if result[0] is None:
                        continue
                    results_0.append(result[0])
                    results_1.append(result[1])
                    results_2.append(result[2])
                except Exception:
                    raise
            #print(results_0[0].shape, results_1[0].shape, results_2[0].shape)
            #print([x.shape for x in results_0])
            return (
                tf.concat(results_0,  axis=0),
                tf.concat(results_1, axis=0),
                tf.concat(results_2, axis=0),
            )
        except Exception:
            return(
                tf.zeros((0, 20, 80, 246), dtype=tf.float32),
                tf.zeros((0, 81, 246), dtype=tf.float32),
                tf.zeros((0, 81, 246), dtype=tf.float32)
            )


    def convert_filepath(self, filepath):
        lines = self.extract_ansi_lines(filepath)
        ansi_array = self.convert(lines, filepath=filepath)
        enc_input, dec_input, labels = TrainingSelector().select_examples(ansi_array)
        return enc_input, dec_input, labels

    def get_lines_from_file(self, filepath):
        try:
            filepath = filepath.numpy().decode('utf-8')
        except Exception as ex:
            raise
        try:
            with open(filepath, "rb") as file:
                #print(file)
                ansi_bytes = file.readlines()
            return [b.decode(encoding, "replace") for b in ansi_bytes]
        except Exception:
            #print("wrong filepath", filepath)
            raise

    def OBSOLETE_get_lines_from_file(self, filepath):
        try:
            #print("this should be a single filepath", filepath)  # If I print here, it really prints a tensor!
            #print(filepath.numpy())
            filepath = filepath.numpy()[0].decode('utf-8')
        except Exception as ex:
            raise
        #print("enter with block")
        with open(filepath, "rb") as file:
            print(file)
            ansi_bytes = file.readlines()
            #print("this should be list of bytes", filepath, len(ansi_bytes), ansi_bytes, type(ansi_bytes[0]))
        #print("with block exited")
        #print([b.decode(encoding, "replace") for b in ansi_bytes])
        return [b.decode(encoding, "replace") for b in ansi_bytes]

    def __init__(self, translator=None):
        self.canvas = DefaultList(self.get_new_blank_line)
        self.modifier_tracker = ModifierTracker()
        self.translator = translator or CharToIntTranslator()
        self.cursor_row = 0
        self.cursor_col = 0
        self.saved_cursor = None

    @property
    def cursor_row(self):
        return self._cursor_row

    @cursor_row.setter
    def cursor_row(self, value):
        if value < 0:
            return
            # raise ValueError("The index value must be non-negative.")
        self._cursor_row = value

    @property
    def cursor_col(self):
        return self._cursor_col

    @cursor_col.setter
    def cursor_col(self, value):
        if not 0 <= value < 80:
            return
            # raise ValueError("The index value must be non-negative.")
        self._cursor_col = value

    def extract_ansi_lines(self, filepath):
        ansi_lines = self.get_lines_from_file(filepath)
        ansi_lines = [line.rstrip("\r\n") for line in ansi_lines]
        return ansi_lines

    def convert(self, ansi_lines, filepath="not set"):
        self.__init__(translator=self.translator)

        esc_m_flagged = False
        remains_of_the_previous_line = ""
        for index, line in enumerate(ansi_lines):

            # if there is some unprocessed ansi sequence, but the file is at the end already
            if remains_of_the_previous_line and not line and index == len(ansi_lines) - 1:
                break

            line = remains_of_the_previous_line + line

            line_at_the_start_of_while_loop = line

            while line:

                if match := re.match(r"\x1b\[(\d+(?:;\d+)*)m(.*)", line):
                    self.modifier_tracker.update(match.groups()[0].split(";"))
                elif match := re.match(r"\x1b\[J(.*)", line):
                    self.delete_the_rest_of_the_line()
                    self.delete_rows_below_the_cursor()
                elif match := re.match(r"\x1b\[2J(.*)", line):
                    self.canvas = DefaultList(self.get_new_blank_line)
                elif match := re.match(r"\x1b\[(\d*);(\d*)[Hft](.*)", line):
                    if match.groups()[0]:
                        self.cursor_row = int(match.groups()[0]) - 1
                    if match.groups()[1]:
                        self.cursor_col = int(match.groups()[1]) - 1
                elif match := re.match(r"\x1b\[(\d+)[Hf](.*)", line):
                    self.cursor_row = int(match.groups()[0]) - 1
                    self.cursor_col = 0
                elif match := re.match(r"\x1b\[[Hf](.*)", line):
                    self.cursor_col = 0
                    self.cursor_row = 0
                elif match := re.match(r"\x1b\[K(.*)", line):
                    self.delete_the_rest_of_the_line()
                elif match := re.match(r"\x1b\[2K(.*)", line):
                    self.canvas[self.cursor_row] = self.get_new_blank_line()
                elif match := re.match(r"\x1b\[\?7h(.*)", line):
                    pass  # basically my implementation already allows line wrapping I THINK
                elif match := re.match(r"\x1b\[\?33l(.*)", line):
                    pass  # I dont know what to do here
                #elif match := re.match(r"\x1b\[m()\Z", line):
                #    pass
                elif match := re.match(r"\x1b\[m(.*)", line):
                    # self.modifier_tracker.update(["0"])
                    if not esc_m_flagged:
                        esc_m_flagged = True
                        raise KnownAnsiSequenceUnimplemented(f"ESC[m in {filepath} {list(line)}")
                elif match := re.match(r"\x1b\[(?:\d+(?:;\d+)?)\x00*()\Z", line):
                    pass
                elif match := re.match(r"\x1b\[(\d*)([ABCD])(.*)", line):
                    for _ in range(int(match.groups()[0] or 1)):
                        if match.groups()[1] == "A":
                            self.cursor_row -= 1
                        if match.groups()[1] == "B":
                            self.cursor_row += 1
                        if match.groups()[1] == "C":
                            self.cursor_col += 1
                        if match.groups()[1] == "D":
                            self.cursor_col -= 1
                elif match := re.match(r"\x1b\[s(?:\\r)?(.*)", line):
                    self.saved_cursor = (self.cursor_row, self.cursor_col)
                elif match := re.match(r"\x1b\[u(.*)", line):
                    self.cursor_row, self.cursor_col = self.saved_cursor
                elif match := re.match(r"\x1b\[M(.*)", line):
                    pass
                elif match := re.match(r"\x1b\[(.*)", line):
                    self.unknown_ansi_sequences.append(list(line))
                    if not any(ansi_lines[index+1:]):
                        break
                    # if there is nothing at the end of this line and the next line empty and is the last line, OK
                    if not remains_of_the_previous_line:
                        remains_of_the_previous_line = line
                        break
                    print(filepath)
                    raise UnknownAnsiSequence(f"Not even wrapping the lines helped smh {list(line)}")
                elif match := re.match(r"^(.*?)((?:\x1b\[.*|$))", line):
                    for char in match.groups()[0]:
                        self.write(char)
       
                else:
                    raise NotImplementedError(f"This ansi sequence is not known {line}")

                try:
                    line = match.groups()[-1]
                except IndexError:
                    raise
                if line and line_at_the_start_of_while_loop == line:
                    raise Exception("While not moving " + line)

                remains_of_the_previous_line = ""  # this rightfully happens at the end of every cycle
            else:
                if line:
                    warning(f"Failed to process everything {line}")

            self.cursor_col = 0
            self.cursor_row += 1
        try:
            return np.array(list(self.canvas))
        except ValueError as ex:
            print(ex)
            raise

    def delete_the_rest_of_the_line(self):
        for col_index in range(self.cursor_col, 80):
            self.canvas[self.cursor_row][col_index] = copy(self.translator.blank_element)

    def delete_rows_below_the_cursor(self):
        for row_index in range(self.cursor_row + 1, len(self.canvas)):
            self.canvas[row_index] = self.get_new_blank_line()

    def write(self, char):
        if ord(char) > 255:
            raise Exception("ord over 255")
        if self.cursor_col >= 80:
            self.cursor_row += 1
            self.cursor_col -= 80

        self.canvas[self.cursor_row][self.cursor_col] = self.translator.translate(char, self.modifier_tracker.modifiers)

        if self.cursor_col == 79:
            self.cursor_col = 0
            self.cursor_row += 1
        else:
            self.cursor_col += 1
