# -*- coding:utf-8 -*-

import copy

class CharArray(object):
    
    def __init__(self, word):
        if not isinstance(word, str):
            raise TypeError("No the correct type for a char array")
        
        self.vocales_fuertes = ["a", "e", "o", "á", "é", "ó", "í", "ú", "ä"]
        self.vocales_debiles = ["i", "u"]
        self.vocales = self.vocales_debiles + self.vocales_fuertes

        self.consonant_y = ["y" + vowel for vowel in self.vocales]
        
        diptongos_crecientes = [d + f for d in self.vocales_debiles\
                                      for f in self.vocales_fuertes]
        diptongos_decrecientes = [f + d for d in self.vocales_debiles\
                                        for f in self.vocales_fuertes] + \
                                 [f + "y" for f in self.vocales_fuertes]
        diptongos_homogeneos = ["iu", "ui"]
        self.diptongos = diptongos_crecientes + \
                         diptongos_decrecientes + \
                         diptongos_homogeneos
        
        self.triptongos = [dip + d for dip in diptongos_crecientes \
                                   for d in self.vocales_debiles] +\
                          [dip + "y" for dip in diptongos_crecientes if \
                                             not dip.endswith("y")]
                
        self.grupos_inseparables = ["br", "cr","dr", "gr", "fr", "kr", "tr", "bl", 
                                    "cl", "gl", "fl", "kl", "pl", "tl", "ll", "ch",
                                    "rr", "pr", "qu", "gü"]
        
        self.word = word
        self.vocal_representation, self.mask_lookup = \
                                        self.build_abstract_representation(word)
        
    def build_abstract_representation(self, word):

        mask_lookup = []

        if word == "y":
            word = "|"
            mask_lookup.append(("V", 0, "y"))

        for consonant_y in self.consonant_y:
            while consonant_y in word:
                index_to_replace = word.index("y")
                word = word.replace("y", "#", 1)
                mask_lookup.append(("C", [index_to_replace,
                                          index_to_replace + 1], "y"))

        for triptongo in self.triptongos:
            while triptongo in word:
                index_to_replace = word.index(triptongo)
                word = word.replace(triptongo, "$$$", 1)
                mask_lookup.append(("V", [index_to_replace,
                                          index_to_replace + 3],
                                    triptongo))

        for diptongo in self.diptongos:
            while diptongo in word:
                index_to_replace = word.index(diptongo)
                word = word.replace(diptongo, "@@", 1)
                mask_lookup.append(("V", [index_to_replace,
                                         index_to_replace + 2],
                                    diptongo))

        for grupo_c in self.grupos_inseparables:
            while grupo_c in word:
                index_to_replace = word.index(grupo_c)
                word = word.replace(grupo_c, "&&", 1)
                mask_lookup.append(("C", [index_to_replace,
                                          index_to_replace + 2],
                                    grupo_c))

        for vowel in self.vocales_debiles + ["y"]:
            while vowel in word:
                index_to_replace = word.index(vowel)
                word = word.replace(vowel, "|", 1)
                mask_lookup.append(("V", [index_to_replace,
                                          index_to_replace + 1], vowel))

        for vowel in self.vocales_fuertes:
            while vowel in word:
                index_to_replace = word.index(vowel)
                word = word.replace(vowel, "¬", 1)
                mask_lookup.append(("V", [index_to_replace,
                                         index_to_replace + 1], vowel))

        for consonant in list("bcdfghjklmnñpqrstvwxz"):
            while consonant in word:
                index_to_replace = word.index(consonant)
                word = word.replace(consonant, "[", 1)
                mask_lookup.append(("C", [index_to_replace,
                                          index_to_replace + 1], 
                                    consonant))
 
        extra_chars = word.replace("|", "").replace("#", "").replace("$", "")\
                .replace("@", "").replace("&", "").replace("¬", "")\
                .replace("[", "")

        if extra_chars != "":
            for extra_char in list(extra_chars):
                while extra_char in word:
                    index_to_replace = word.index(extra_char)
                    word = word.replace(extra_char, "]", 1)
                    mask_lookup.append(("C", [index_to_replace,
                                              index_to_replace + 1],
                                        extra_char))
        #   | -> y_vocal, vocal debil
        #   # -> y_consonante
        # $$$ -> triptongo
        #  @@ -> diptongo
        #  && -> grupo consonantico
        #  ¬ -> vocal fuerte
        #  [ -> consonante
        #  ] -> otros caracteres (considerados consonante)
        word = word.replace("|", "V").replace("#", "C").replace("$$$", "V")\
                .replace("@@", "V").replace("&&", "C").replace("¬", "V")\
                .replace("[", "C").replace("]", "C")
        mask_lookup = sorted(mask_lookup, key=lambda elem: elem[1])
        return word, mask_lookup

    def unmask(self, pattern):
        result = []
        syllabic_pattern = []
        mask_lookup = copy.copy(self.mask_lookup)
        for syllable in pattern:
            subsyl = ""
            subsylpattern = ""
            for character in syllable:
                remembered_char, interval, corresponding_string = \
                                                            mask_lookup.pop(0)
                if character == remembered_char:
                    subsyl += corresponding_string
                    subsylpattern += remembered_char * len(corresponding_string)
                else:
                    raise ValueError("The pattern doesn't correspond to this\
                                     word: %s" % self.word)
            result.append(subsyl)
            syllabic_pattern.append(subsylpattern)
        return result, syllabic_pattern

    def __str__(self, *args, **kwargs):
        return str(self.vocal_representation)
    
    def __repr__(self, *args, **kwargs):
        return str(self)
    
    
class Silabicador(object):
    
    def __call__(self, word):
        '''http://ponce.inter.edu/acad/cursos/ciencia/lasvi/modulo2.htm'''
        res = []
        lower_word = word.lower()
        char_array = CharArray(lower_word)
        abstract_word = list(str(char_array))
        while len(abstract_word) != 0:
            if abstract_word[0] == "V":
                if len(abstract_word) == 1:
                    res += ["V"]
                    abstract_word = []
                elif len(abstract_word) == 2:
                    if abstract_word[1] == "C":
                        res += ["VC"]
                    else:
                        res += ["V", "V"]
                    abstract_word = []
                elif len(abstract_word) == 3:
                    res += ["V", abstract_word[1] + abstract_word[2]]
                    abstract_word = []
                else:
                    # No hay consonantes en frente, sino otra vocal
                    if abstract_word[1] == "V":
                        res += ["V"]
                        del abstract_word[0]
                    # Una consonante entre dos vocales se agrupa con la vocal de la derecha:
                    elif abstract_word[1] == "C" and\
                       abstract_word[2] == "V":
                        res += ["V", "CV"]
                        del abstract_word[2]
                        del abstract_word[1]
                        del abstract_word[0]
                    # Dos consonantes entre dos vocales se separan y cada consonante se queda con una vocal:
                    elif abstract_word[1] == "C" and\
                       abstract_word[2] == "C" and\
                       abstract_word[3] == "V":
                        res += ["VC", "CV"]
                        del abstract_word[3]
                        del abstract_word[2]
                        del abstract_word[1]
                        del abstract_word[0]

                    # Cuando hay tres consonantes entre vocales, las primeras dos se unen con la primera vocal y la tercera se une a la segunda vocal.
                    elif len(abstract_word) > 4 and\
                         abstract_word[1] == "C" and\
                         abstract_word[2] == "C" and\
                         abstract_word[3] == "C" and\
                         abstract_word[4] == "V":
                        res += ["VCC", "CV"]
                        del abstract_word[4]
                        del abstract_word[3]
                        del abstract_word[2]
                        del abstract_word[1]
                        del abstract_word[0]
                    # Cuando hay cuatro consonantes entre vocales, las primeras dos se unen a la primera vocal y las otras dos se unen a la segunda vocal.
                    elif len(abstract_word) > 5 and\
                         abstract_word[1] == "C" and\
                         abstract_word[2] == "C" and\
                         abstract_word[3] == "C" and\
                         abstract_word[4] == "C" and\
                         abstract_word[5] == "V":
                        res += ["VCC", "CCV"]
                        del abstract_word[5]
                        del abstract_word[4]
                        del abstract_word[3]
                        del abstract_word[2]
                        del abstract_word[1]
                        del abstract_word[0]
                    # Chain of consonants
                    else:
                        count = 0
                        index = 1
                        final_letter = ""
                        while index < len(abstract_word) and abstract_word[index] != "V":
                            final_letter = abstract_word[index]
                            index += 1
                            count += 1
                        if index == len(abstract_word):
                            index -= 1
                        if final_letter == "V":
                            res += ["VCC", "C"*(count-4), "CCV"]
                        else:
                            res += ["VCC", "C"*(count-2)]
                        for i in range(index, -1, -1):
                            del abstract_word[i]
            
            elif abstract_word[0] == "C":
                res.append(abstract_word.pop(0))
        
        final_grouping = []
        while len(res)>0:
            if res[0] == "C" and \
               len(res) != 1 and \
               res[1].startswith("V"):
                final_grouping.append(res[0] + res[1])
                del res[1]
                del res[0]
            # Si existe la consonante pega con la silaba anterior
            elif res[0] == "C" and\
                 len(final_grouping)>0 and\
                 final_grouping[-1].endswith("V"): 
                final_grouping[-1] = final_grouping[-1] + res[0]
                del res[0]
            # Else, assume it is a valid syllable
            else:
                final_grouping.append(res[0])
                del res[0]

        return char_array.unmask(final_grouping)
