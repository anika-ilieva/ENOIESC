from cmath import exp
from thesis.models.Sentence import Sentence
from thesis.helpers.OffsetHelper import OffsetHelper
import unittest
import spacy
import random
import os
import sys
import pathlib

TOP_LEVEL_DIRECTORY = str(pathlib.Path(__file__).parent.resolve().parent.absolute())
sys.path.insert(0, TOP_LEVEL_DIRECTORY)


class SentenceMethods(unittest.TestCase):
  def test_sentence(self):
    text = "This is an example sentence to show that the sentence method works correctly."
    sent = Sentence(text)
    assert sent.sentence() == text

  def test_words(self):
    text = "This is an example sentence to show that the words method works correctly."
    expected = ["This", "is", "an", "example", "sentence", "to",
                "show", "that", "the", "words", "method", "works", "correctly."]

    sent = Sentence(text)
    assert sent.words() == expected

    text = "This is an example sentence with commas and A, B, C."
    expected = ["This", "is", "an", "example", "sentence",
                "with", "commas", "and", "A,", "B,", "C."]

    sent = Sentence(text)
    assert sent.words() == expected

  def test_sent_doc(self):
    text = "This is an example sentence to show that the sent_doc method works correctly."
    sent = Sentence(text)
    assert sent.parsable()

    assert type(sent.sent_doc()) == spacy.tokens.doc.Doc

  def test_conj_expr_overlaps(self):
    sent = Sentence('')
    sent._conj_term_offsets = [[[[0,1]],[[2,3]]], [[[4,5]], [[6,7]]]]
    assert sent.conj_expr_overlaps() == []
    
    sent = Sentence('')
    sent._conj_term_offsets = [[[[0, 1]], [[2, 3]]], [[[2, 3]], [[6, 7]]]]
    assert sent.conj_expr_overlaps() == [[0, 1]]

  def test_conj_term_offsets(self):
    text = "This sentence has a conjunction with Apple, Bee and a happy Clown, but it does not overlap the conjunction: Dog, Elephant and Fox."
    expected_offsets = [
      [
        [[37, 41]],
        [[44, 46]],
        [[60, 64]]
      ],
      [
        [[108, 110]],
        [[113, 120]],
        [[126, 128]]
      ]
    ]

    sent = Sentence(text)
    assert sent.conj_term_offsets() == expected_offsets

  def test_num_of_conj_exprs(self):
    text = "This sentence has a conjunction with Apple, Bee and a happy Clown, but it does not overlap the conjunction: Dog, Elephant and Fox."
    sent = Sentence(text)
    assert sent.num_of_conj_exprs() == 2

  def test_num_of_conj_terms_per_conj_expr(self):
    text = "This sentence has a conjunction with Apple, Bee and a happy Clown, but it does not overlap the conjunction: Dog, Elephant and Fox."
    sent = Sentence(text)
    assert sent.num_of_conj_terms_per_conj_expr() == [3, 3]

  def test_conj_offsets(self):
    text = "This sentence has a conjunction with Apple, Bee and a happy Clown, but it does not overlap the conjunction: Dog, Elephant and Fox."
    expected_offsets = [
      [37, 64],
      [108, 128]
    ]

    sent = Sentence(text)
    assert sent.conj_expr_offsets() == expected_offsets

  def test_conj_debug(self):
    text = "This sentence has a conjunction with Apple, Bee and a happy Clown, but it does not overlap the conjunction: Dog, Elephant and Fox."

    expected_conj_string = "(Apple) (Bee) (Clown) | (Dog) (Elephant) (Fox)"

    sent = Sentence(text)
    assert sent.conj_debug() == (text, expected_conj_string), sent.conj_debug()
  
  def test_conj_debug_with_supplements(self):
    text = "This sentence has a conjunction with Apple, Bee and a happy Clown, but it does not overlap the conjunction: Dog, Elephant and Fox."

    expected_conj_string = "(Apple) (Bee) (a happy Clown) | (Dog) (Elephant) (Fox)"

    sent = Sentence(text)
    assert sent.conj_debug_with_supplements() == (text, expected_conj_string), sent.conj_debug()

  def test_sentence_pieces(self):
    text = "Peter likes apples, blue bananas from chile and cherries."
    sent = Sentence(text)

    assert sent.sentence_pieces() == ['Peter',
                                      'likes',
                                      'apples',
                                      ',',
                                      'blue',
                                      'banana',
                                      '##s',
                                      'from',
                                      'ch',
                                      '##ile',
                                      'and',
                                      'ch',
                                      '##erries',
                                      '.']

  def test_sentence_pieces_offset_map(self):
    text = "Peter likes apples, blue bananas from chile and cherries."
    sent = Sentence(text)

    assert sent.sentence_pieces_offset_map() == [[0, 4],
                                                 [6, 10],
                                                 [12, 17],
                                                 [18, 18],
                                                 [20, 23],
                                                 [25, 30],
                                                 [31, 31],
                                                 [33, 36],
                                                 [38, 39],
                                                 [40, 42],
                                                 [44, 46],
                                                 [48, 49],
                                                 [50, 55],
                                                 [56, 56]]

  def test_offsets_to_sentence_pieces(self):
    text = "Peter likes apples, blue bananas from chile and cherries."
    sent = Sentence(text)

    # word to pieces mapping
    expected = [[0,0], [1,1], [2,3], [4,4], [5,6], [7,7], [8,9], [10,10], [11,13]]

    offset = 0
    for word_idx, word in enumerate(sent.words()):
      word_offset = [offset, offset + len(word) - 1]

      assert sent.offsets_to_sentence_pieces([word_offset]) == [expected[word_idx]]
      offset += len(word) + 1

    # check comma
    assert sent.offsets_to_sentence_pieces([[18, 18]]) == [[3,3]]

  def test_conjunct_offsets(self):
    text = "This sentence has a conjunction with Apple, Bee and a happy Clown, but it does not overlap the conjunction: Dog, Elephant and Fox."
    sent = Sentence(text)
    offsets = sent.conjunction_offsets()

    assert offsets == [[[[42, 42]], [[48, 50]]], [[[111, 111]], [[122, 124]]]], offsets
    assert [[sent.offsets_to_strings(offset) for offset in conj] for conj in offsets] == [
        [[','], ['and']], [[','], ['and']]]

if __name__ == '__main__':
  unittest.main()
