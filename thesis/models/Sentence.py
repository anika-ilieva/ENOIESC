
from transformers import BertTokenizerFast
from thesis.helpers.OffsetHelper import OffsetHelper
from thesis.helpers.LabelHelper import LabelHelper
from thesis.helpers.ConjTermSupplementer import ConjTermSupplementer
import spacy
from spacy import displacy
import en_core_web_trf
nlp = en_core_web_trf.load()
nlp.add_pipe("merge_entities")

BERT_TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-cased')

"""Terminology

conjunction: Represents a linguistic form that joins together sentences, clauses, phrases, or words like "and", "or", commas, etc.
  Example: "Obama visited Inida, China and Japan."
    The conjunctions are "," and "and"

conjunction term: Represents the words joined by a conjunction.
  Example: "Obama visited Inida, China and Japan.":
    The conjunction terms in the example sentence are "India", "China" and "Japan"

conjunction term supplement: Represents a conjunction term with additional information that describes the conjunction term.
  Example: "Obama visited fascinating Inida, astonishing China and charming Japan."
    The conjunction terms are "India", "China" and "Japan"
    The supplemented conjunction terms are "fascinating Inida", "astonishing China" and "charming Japan"

conjunction expression: Represents the whole phrase from the first to the last conjunction term.
  Example: "Obama visited Inida, China and Japan.":
    The first conjunction expression is "Inida, China and Japan"

offset (also offset interval): represents an interval [begin, end] of character indices
  Example: "This is an example sentence.":
    The offset interval of the word "example" would be [11, 17].
    Notice you would get the corresponding string by slicing: sentence[begin:end+1]

pos (also position interval): represents an interval [begin, end] of word indices
  Example: "This is an example sentence.":
    The pos interval of the word "example" would be [3, 3].
    The pos interval of "an example" would be [2, 3].

"""


class Sentence:
  def __init__(self, sentence) -> None:
    self._sentence = sentence.replace('\xa0', ' ')
    self._sent_doc = None
    self._parsable = None

    # spacy
    self._conj_ends = None
    self._conj_heads = None
    self._sorted_conj_head_tokens = None

    self._conj_expr_offsets = None
    self._conj_expr_offsets_with_supplements = None
    self._num_of_conj_exprs = None
    self._head_cc_tokens = None
    self._cc_tokens = None
    self._cc_offsets = None
    self._conjunction_comma_offsets = None
    self._conjunction_comma_tokens = None
    self._conjunction_offsets = None
    self._conj_tokens = None
    self._conj_tokens_with_supplements = None
    self._conj_supplements = None
    self._conj_term_offsets = None
    self._conj_term_offsets_with_supplements = None
    self._num_of_conj_terms_per_conj_expr = None
    self._conj_expr_overlaps = None
    self._conj_expr_overlaps_with_supplements = None
    self._conj_term_overlaps_with_supplements = None
    self._words = None
    self._sentence_pieces = None
    self._word2piece = None
    self._sentence_pieces_offset_map = None
    self._reversed_sentence_pieces_offset_map = None

    self._spacy_parser = nlp

  def sentence(self):
    return self._sentence

  def words(self):
    if self._words is None:
      self._words = self.sentence().split(' ')
    return self._words

  def word2piece(self):
    if self._word2piece is None:
      self.sentence_pieces()
    return self._word2piece

  def sentence_pieces(self):
    if self._sentence_pieces is None:
      word2piece = {idx: list() for idx in range(len(self.words()))}
      sentence_pieces = list()
      piece_idx = 0
      for word_idx, word in enumerate(self.words()):
        pieces = BERT_TOKENIZER.tokenize(word)
        sentence_pieces += pieces
        for piece_idx_added, piece in enumerate(pieces):
          word2piece[word_idx].append(piece_idx + piece_idx_added)
        piece_idx += len(pieces)
      assert len(sentence_pieces) == piece_idx
      self._word2piece = word2piece
      self._sentence_pieces = sentence_pieces

    return self._sentence_pieces

  def sentence_pieces_offset_map(self):
    if self._sentence_pieces_offset_map is None:
      tokens = BERT_TOKENIZER.encode_plus(
        self.sentence(), return_offsets_mapping=True)
      self._sentence_pieces_offset_map = [[begin, end - 1]
                                          for (begin, end) in tokens['offset_mapping'][1:-1]]

    return self._sentence_pieces_offset_map

  def reversed_sentence_pieces_offset_map(self):
    if self._reversed_sentence_pieces_offset_map is None:
      self._reversed_sentence_pieces_offset_map = {}
      for piece_idx, offset in enumerate(self.sentence_pieces_offset_map()):
        for offset_idx in range(offset[0], offset[1] + 1):
          self._reversed_sentence_pieces_offset_map[offset_idx] = piece_idx
    return self._reversed_sentence_pieces_offset_map

  def offsets_to_sentence_pieces(self, offsets):
    combined_offsets = OffsetHelper.combine_intervals(offsets)
    reversed_word_offset_map = self.reversed_sentence_pieces_offset_map()

    result = []

    for offset in combined_offsets:
      piece_ids = []

      for idx in range(offset[0], offset[1] + 1):
        if self.sentence()[idx] != ' ':
          piece_ids.append([reversed_word_offset_map[idx],
                           reversed_word_offset_map[idx]])
      result += OffsetHelper.combine_intervals(piece_ids)

    return result

  def pred_label_to_string(self, label, shift=0):
    return LabelHelper.pred_label_to_string(label, self, shift)

  def arg_label_to_string(self, label, all_conjunction_idxs=[], shift=0):
    return LabelHelper.arg_label_to_string(label, self, all_conjunction_idxs, shift)

  def all_conjunction_label_to_string(self, label):
    return LabelHelper.all_conjunction_label_to_string(label, self)

  def all_conj_expr_label_to_string(self, label):
    return LabelHelper.all_conj_expr_label_to_string(label, self)

  def all_conj_term_label_to_string(self, label):
    return LabelHelper.all_conj_term_label_to_string(label, self)

  def sent_doc(self) -> spacy.tokens.doc.Doc:
    if self._sent_doc is None:
      self._parsable = False
      try:
        self._sent_doc = self._spacy_parser(self._sentence)
        self._parsable = True
      except:
        return None
    return self._sent_doc

  def displacy(self):
    displacy.serve(self.sent_doc(), style="dep")

  def check_parsable(self) -> None:
    assert self.parsable(), "Sentence is not parsable"

  def parsable(self) -> bool:
    if self._parsable is None:
      self.sent_doc()
    return self._parsable

  def conj_ends(self) -> list:
    self.check_parsable()

    if self._conj_ends is None:
      self._conj_ends = [token for token in self.sent_doc() if (token.dep_ == "conj") and "conj" not in [
          child.dep_ for child in token.children] and (token.pos_ != 'VERB') and (token.head.pos_ != 'VERB')]
    return self._conj_ends

  def conj_heads(self) -> list:
    if self._conj_heads is None or self._head_cc_tokens is None:
      self._conj_heads = {}
      self._head_cc_tokens = {}
      for conj_end in self.conj_ends():
        conj_list = [conj_end]
        cc_list = []
        conj = conj_end

        skip = False
        while not skip and conj.dep_ == "conj":
          assert conj.pos_ != "VERB", (self.sentence(), conj)
          for child in conj.children:
            if child.dep_ == 'cc':
              if str(child) in ['or', 'and', ',', 'but', 'and/or', 'as', 'nor']:
                cc_list.append(child)
              else:
                skip = True
                break

          if conj.head.pos_ != "VERB":
            conj_list.append(conj.head)
            conj = conj.head
          else:
            break
       
        for child in conj.children:
          if child.dep_ == 'cc':
            if str(child) in ['or', 'and', ',', 'but', 'and/or', 'as', 'nor']:
              cc_list.append(child)
            else:
              skip = True
              break
        if skip:
          continue
        if conj not in self._conj_heads:
          self._conj_heads[conj] = []
        if conj not in self._head_cc_tokens:
          self._head_cc_tokens[conj] = []
        self._conj_heads[conj] += conj_list
        self._head_cc_tokens[conj] += cc_list

      self._conj_heads = {k: sorted(list(set(v)), key=lambda x: x.idx)
                               for k, v in self._conj_heads.items()}
      self._head_cc_tokens = {k: sorted(
        list(set(v)), key=lambda x: x.idx) for k, v in self._head_cc_tokens.items()}
    return self._conj_heads

  def head_cc_tokens(self):
    self.conj_heads()
    return self._head_cc_tokens

  def sorted_conj_head_tokens(self):
    if self._sorted_conj_head_tokens is None:
      self._sorted_conj_head_tokens = sorted(
          list(self.conj_heads().keys()), key=lambda x: x.idx)
    return self._sorted_conj_head_tokens

  def conj_tokens(self) -> list:
    if self._conj_tokens is None:
      self._conj_tokens = [self.conj_heads()[token]
                           for token in self.sorted_conj_head_tokens()]
    return self._conj_tokens

  def conjunction_comma_tokens(self) -> list:
    if self._conjunction_comma_tokens is None:
      self._conjunction_comma_tokens = []

      for conj in self.conj_tokens_with_supplements():
        comma_list = []
        for conj_term in conj[:-1]:
          for token in conj_term:
            if token.n_rights > 0:
              nbor = token.nbor()
              if str(nbor) == ',':
                comma_list.append(nbor)
        self._conjunction_comma_tokens.append(comma_list)
    return self._conjunction_comma_tokens

  def conjunction_comma_offsets(self) -> list:
    if self._conjunction_comma_offsets is None:
      self._conjunction_comma_offsets = [
        [[[token.idx, token.idx + len(token) - 1]] for token in conj] for conj in self.conjunction_comma_tokens()]
    return self._conjunction_comma_offsets

  def cc_tokens(self) -> list:
    if self._cc_tokens is None:
      self.conj_heads()
      self._cc_tokens = []
      for head in self.sorted_conj_head_tokens():
        conj_cc_list = []
        for token in self.head_cc_tokens()[head]:
          token_list = [token]
          for token in token.children:
            if token.dep_ == "advmod":
              token_list.append(token)
          conj_cc_list.append(sorted(token_list, key=lambda x: x.idx))
        self._cc_tokens.append(conj_cc_list)
    return self._cc_tokens

  def cc_offsets(self) -> list:
    if self._cc_offsets is None:
      self._cc_offsets = [
        [[[token.idx, token.idx + len(token) - 1] for token in cc] for cc in conj]for conj in self.cc_tokens()]
    return self._cc_offsets

  def conjunction_offsets(self) -> list:
    if self._conjunction_offsets is None:
      assert len(self.cc_offsets()) == len(self.conjunction_comma_offsets())
      conjunction_offsets = []
      for cc_offsets, conjunction_comma_offsets in zip(self.cc_offsets(), self.conjunction_comma_offsets()):
        conjunction_offsets.append(
          sorted((cc_offsets + conjunction_comma_offsets), key=lambda offset: offset[0]))
      self._conjunction_offsets = conjunction_offsets
    return self._conjunction_offsets

  def conj_term_offsets(self) -> list:
    """Returns the offsets of the conjunction terms.

    Offsets represent character indices.
    Offset interval: [begin, end] represents "String"[begin:end+1]

    """
    if self._conj_term_offsets is None:
      conj_term_offsets = []

      for conj_term_tokens in self.conj_tokens():
        cur_conj_term_offsets = [
          [[token.idx, token.idx + len(token) - 1]] for token in conj_term_tokens]

        conj_term_offsets.append(cur_conj_term_offsets)
      self._conj_term_offsets = conj_term_offsets
    return self._conj_term_offsets

  def conj_term_offsets_with_supplements(self) -> list:
    """Returns the offsets of the conjunction terms.

    Offsets represent character indices.
    Offset interval: [begin, end] represents "String"[begin:end+1]

    """
    if self._conj_term_offsets_with_supplements is None:
      conj_term_offsets = []

      for conj_tokens in self.conj_tokens_with_supplements():
        cur_conj_term_offsets = [[[conj_term_token.idx, conj_term_token.idx + len(
          conj_term_token) - 1] for conj_term_token in conj_terms] for conj_terms in conj_tokens]

        conj_term_offsets.append(cur_conj_term_offsets)
      self._conj_term_offsets_with_supplements = conj_term_offsets
    return self._conj_term_offsets_with_supplements

  def num_of_conj_terms_per_conj_expr(self) -> int:
    if self._num_of_conj_terms_per_conj_expr is None:
      self._num_of_conj_terms_per_conj_expr = [
        len(conj_terms) for conj_terms in self.conj_term_offsets()]
    return self._num_of_conj_terms_per_conj_expr

  def conj_expr_offsets(self) -> list:
    """ Returns the intervals of the conjunction expressions (begin of the first conjunction term and end of the last)
    """

    if self._conj_expr_offsets is None:
      self._conj_expr_offsets = [[conj[0][0][0], conj[-1][-1][1]]
                            for conj in self.conj_term_offsets()]
    return self._conj_expr_offsets

  def conj_expr_offsets_with_supplements(self) -> list:
    """ Returns the intervals of the conjunction expressions (begin of the first conjunction term and end of the last)
    """

    if self._conj_expr_offsets_with_supplements is None:
      self._conj_expr_offsets_with_supplements = [[conj[0][0][0], conj[-1][-1][1]]
                                             for conj in self.conj_term_offsets_with_supplements()]
    return self._conj_expr_offsets_with_supplements

  def num_of_conj_exprs(self) -> int:
    if self._num_of_conj_exprs is None:
      self._num_of_conj_exprs = len(self.conj_expr_offsets())
    return self._num_of_conj_exprs

  def conj_debug(self) -> tuple:
    conj_term_offsets = self.conj_term_offsets()
    conj_string = ' | '.join([' '.join(['(' + ' '.join(
        [self._sentence[begin:end + 1] for begin, end in conj_term]) + ')' for conj_term in conj]) for conj in conj_term_offsets])
    return self._sentence, conj_string

  def conj_debug_with_supplements(self) -> tuple:
    conj_term_offsets = self.conj_term_offsets_with_supplements()
    conj_string = ' | '.join([' '.join(['(' + ' '.join(
        [self._sentence[begin:end + 1] for begin, end in conj_term]) + ')' for conj_term in conj]) for conj in conj_term_offsets])
    return self._sentence, conj_string

  def conj_expr_overlaps(self) -> list:
    if self._conj_expr_overlaps is None:
      overlaps = []

      for i, conj_expr_a in enumerate(self.conj_expr_offsets()):
        for j, conj_expr_b in enumerate(self.conj_expr_offsets()[i + 1:]):
          if OffsetHelper.intervals_overlap(conj_expr_a, conj_expr_b):
            overlaps.append([i, j + i + 1])
      self._conj_expr_overlaps = overlaps

    return self._conj_expr_overlaps

  def conj_term_overlaps_with_supplements(self) -> list:
    if self._conj_term_overlaps_with_supplements is None:
      self._conj_term_overlaps_with_supplements = []
      conj_term_intervals = [[[conj_term[0][0], conj_term[-1][1]]
                              for conj_term in conj] for conj in self.conj_term_offsets_with_supplements()]

      for conj_id, conj_terms in enumerate(conj_term_intervals):
        if OffsetHelper.intervals_cross_overlap(conj_terms):
          self._conj_term_overlaps_with_supplements.append(conj_id)
    return self._conj_term_overlaps_with_supplements

  def conj_expr_overlaps_with_supplements(self) -> list:
    if self._conj_expr_overlaps_with_supplements is None:
      overlaps = []

      for i, conj_expr_a in enumerate(self.conj_expr_offsets_with_supplements()):
        for j, conj_expr_b in enumerate(self.conj_expr_offsets_with_supplements()[i + 1:]):
          if OffsetHelper.intervals_overlap(conj_expr_a, conj_expr_b):
            overlaps.append([i, j + i + 1])
      self._conj_expr_overlaps_with_supplements = overlaps

    return self._conj_expr_overlaps_with_supplements

  def offsets_to_strings(self, offsets):
    sentence = self.sentence()
    return [sentence[begin:end + 1] for begin, end in offsets]

  def conj_tokens_with_supplements(self):
    if self._conj_tokens_with_supplements is None:
      self._conj_tokens_with_supplements = []

      conj_tokens = self.conj_tokens()

      for conj in conj_tokens:
        conj_result = []
        for conj_term in conj:
            # get additional words
          supplements = ConjTermSupplementer.get_conj_term_supplements(
            conj_term)
          supplemented_conj_term = supplements + [conj_term]
          supplemented_conj_term = sorted(
            supplemented_conj_term, key=lambda token: token.idx)

          conj_result.append(supplemented_conj_term)
        self._conj_tokens_with_supplements.append(conj_result)
    return self._conj_tokens_with_supplements

  def conj_supplements(self):
    if self._conj_supplements is None:
      self._conj_supplements = []

      conj_tokens = self.conj_tokens()

      for conj in conj_tokens:
        conj_result = []
        for conj_term in conj:
            # get additional words
          supplements = ConjTermSupplementer.get_conj_term_supplements(
            conj_term)
          supplemented_conj_term = supplements
          supplemented_conj_term = sorted(
            supplemented_conj_term, key=lambda token: token.idx)

          conj_result.append(supplemented_conj_term)
        self._conj_supplements.append(conj_result)
    return self._conj_supplements
