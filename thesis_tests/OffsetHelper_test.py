from thesis.helpers.OffsetHelper import OffsetHelper
import unittest
import os
import sys
import pathlib

TOP_LEVEL_DIRECTORY = str(pathlib.Path(__file__).parent.resolve().parent.absolute())
sys.path.insert(0, TOP_LEVEL_DIRECTORY)


class TestOffsetHelperMethods(unittest.TestCase):
  def test_intervals_overlap(self):
    positive_examples = [
      [[1, 2], [2, 3]],  # a overlaps b
      [[1, 9], [2, 8]],  # a contains b
      [[1, 9], [1, 9]]  # a is equal to b
    ]

    negative_examples = [
      [[1, 2], [3, 4]],  # a only touches b
      [[1, 2], [4, 5]]  # a does not touch/overlap b
    ]

    for example in positive_examples:
      assert OffsetHelper.intervals_overlap(example[0], example[1]), example
      assert OffsetHelper.intervals_overlap(
        example[1], example[0]), example[::-1]

    for example in negative_examples:
      assert not OffsetHelper.intervals_overlap(
        example[0], example[1]), example
      assert not OffsetHelper.intervals_overlap(
        example[1], example[0]), example[::-1]

  def test_combine_intervals(self):
    examples = [
      [[[1, 2], [3, 4]], [[1, 4]]],  # a touches b
      [[[1, 3], [3, 4]], [[1, 4]]],  # a overlaps with b
      # a overlaps with b (more than one unit overlap)
      [[[1, 3], [2, 4]], [[1, 4]]],
      [[[1, 3], [1, 4]], [[1, 4]]],  # b contains a

      [[[1, 2], [3, 4], [5, 6]], [[1, 6]]],  # chain of touching intervals
      [[[1, 2], [2, 3], [3, 4]], [[1, 4]]],  # chain of overlapping intervals
      # chain of overlapping intervals (more than one unit overlap)
      [[[1, 3], [2, 4], [3, 5]], [[1, 5]]],

      # chain of containing intervals with order ABC
      [[[1, 3], [1, 6], [1, 9]], [[1, 9]]],
      # chain of containing intervals with order ACB
      [[[1, 3], [1, 9], [1, 6]], [[1, 9]]],
      # chain of containing intervals with order BAC
      [[[1, 6], [1, 3], [1, 9]], [[1, 9]]],
      # chain of containing intervals with with order BCA
      [[[1, 6], [1, 9], [1, 3]], [[1, 9]]],
      # chain of containing intervals with with order CAB
      [[[1, 9], [1, 3], [1, 6]], [[1, 9]]],
      # chain of containing intervals with with order CBA
      [[[1, 9], [1, 6], [1, 3]], [[1, 9]]],

      [[[1, 5], [4, 9], [20, 22], [24, 25], [26, 27], [27, 28], [30, 35], [38, 40], [
        38, 45]], [[1, 9], [20, 22], [24, 28], [30, 35], [38, 45]]]  # mixed cases
    ]

    for input, result in examples:
      assert OffsetHelper.combine_intervals(input) == result, (input, result)

    with self.assertRaisesRegex(AssertionError, 'Interval array must be sorted'):
      OffsetHelper.combine_intervals([[2, 3], [1, 3]])

if __name__ == '__main__':
  unittest.main()
