import numpy


class CharToIntTranslator:
    blank_element = ord(" ")

    @classmethod
    def translate(cls, char, modifiers):
        encoded_modifiers = 0
        foreground_modifier_found = False

        for code in map(str, modifiers):
            if code == "1":
                encoded_modifiers += 64  # this is the seventh bit
            elif code == "5":
                encoded_modifiers += 128  # this is the eighth bit
            elif code[0] == "3":  # this is the first three bits
                encoded_modifiers += int(code[1])
                foreground_modifier_found = True
            elif code[0] == "4":
                encoded_modifiers += int(code[1]) * 8  # this is the fourth, fifth and sixth bit
            else:
                print(f"Value not expected {code}")

        if not foreground_modifier_found:
            # if there is no specified color for foreground, default to white
            encoded_modifiers += 7

        return ord(char) + 256 * encoded_modifiers

    @classmethod
    def translate_back(cls, integer):
        char = chr(integer % 256)
        encoded_modifiers = integer // 256
        modifiers = set()
        if encoded_modifiers & 128:
            modifiers.add("5")
        if encoded_modifiers & 64:
            modifiers.add("1")

        bg_code = (encoded_modifiers & int("111000", 2)) // int("1000", 2)
        modifiers.add(f"4{bg_code}")
        fg_code = encoded_modifiers & int("111", 2)
        modifiers.add(f"3{fg_code}")

        return char, modifiers


class CharToVecTranslator:
    blank_element = numpy.zeros(264)
    blank_element[32] = 1
    rgb_map = {
        "0": (0, 0, 0),
        "1": (1, 0, 0),
        "2": (0, 1, 0),
        "3": (1, 1, 0),
        "4": (0, 0, 1),
        "5": (1, 0, 1),
        "6": (0, 1, 1),
        "7": (1, 1, 1),
    }
    inverese_rgb_map = {value: key for key, value in rgb_map.items()}

    def translate(self, char, modifiers):
        encoded_modifiers = 0
        foreground_modifier_found = False

        resulting_vector = numpy.zeros(264)
        resulting_vector[ord(char)] = 1
        try:
            for code in map(str, modifiers):
                if code == "1":
                    resulting_vector[256] = 1  # this is the seventh bit
                elif code == "5":
                    resulting_vector[260] = 1  # this is the eighth bit
                elif code[0] == "3":
                    foreground_modifier_found = True
                    resulting_vector[range(257, 260)] = self.rgb_map[code[1]]
                elif code[0] == "4":
                    resulting_vector[range(261, 264)] = self.rgb_map[code[1]]
                else:
                    print(f"Value not expected {code}")
        except KeyError:
            print(char, modifiers)
            raise

        if not foreground_modifier_found:
            # if there is no specified color for foreground, default to white
            resulting_vector[range(257, 260)] = (1, 1, 1)

        return resulting_vector

    def translate_back(self, vector):
        modifiers = []
        char_data = vector[:256]
        char_int = int(numpy.dot(char_data, numpy.array(range(256))))
        fg_bold = vector[256]
        fg_color = tuple(vector[257:260])
        bg_bold = vector[260]
        bg_color = tuple(vector[261:264])

        if fg_bold:
            modifiers.append("1")
        if bg_bold:
            modifiers.append("5")
        modifiers.append("3" + self.inverese_rgb_map[fg_color])
        modifiers.append("4" + self.inverese_rgb_map[bg_color])
        print(char_int)
        if char_int == 10:
            char_int = 32  # turns out \n i.e. chr(10) messes with the output, so this is a hotfix
        return chr(char_int), modifiers


class ModifierTracker:

    def __init__(self):
        self.modifier_dict = dict()

    @property
    def modifiers(self):
        return set(self.modifier_dict.values())

    def update(self, new_modifiers):
        new_modifiers = set(map(str, new_modifiers))

        if "0" in new_modifiers:
            self.modifier_dict = dict()
            new_modifiers.remove("0")

        for code in new_modifiers:
            if code == "1":
                self.modifier_dict["fg_bright"] = "1"
            elif code == "5":
                self.modifier_dict["bg_bright"] = "5"
            elif code[0] == "3":  # this is the first three bits
                self.modifier_dict["fg_color"] = code
            elif code[0] == "4":
                self.modifier_dict["bg_color"] = code

        return self.modifiers
