import numpy as np

from translators.translation_utils import ModifierTracker, CharToIntTranslator, CharToVecTranslator


class AnsFromNumConverter:
    @staticmethod
    def convert(int_array: np.array):
        file_content = []
        last_modifiers = None
        for row in np.nditer(int_array, flags=['external_loop'], order='F'):
            for integer in row:
                char, modifiers = CharToVecTranslator.translate_back(integer)
                if modifiers != last_modifiers:
                    file_content.append("\x1b[" + ";".join(["0"] + [str(mod) for mod in modifiers]) + "m")
                    last_modifiers = modifiers
                file_content.append(char)

        return "".join(file_content)

class AnsFromVecConverter:
    @staticmethod
    def convert(int_array: np.array):
        file_content = []
        last_modifiers = None
        for row in range(int_array.shape[0]):
            for col in range(int_array.shape[1]):
                char, modifiers = CharToVecTranslator().translate_back(int_array[row, col, :])
                if modifiers != last_modifiers:
                    file_content.append("\x1b[" + ";".join(["0"] + [str(mod) for mod in modifiers]) + "m")
                    last_modifiers = modifiers
                file_content.append(char)

        return "".join(file_content)
