import itertools
import numpy as np


class OffsetHelper():
  @staticmethod
  def intervals_cross_overlap(intervals):
    for i, interval_a in enumerate(intervals):
      for interval_b in intervals[i + 1:]:
        if OffsetHelper.intervals_overlap(interval_a, interval_b):
          return True
    return False

  @staticmethod
  def interval_is_in(a: list, b: list):
    assert len(a) == 2
    assert len(b) == 2
    assert type(a[0]) == int
    assert type(a[1]) == int
    assert type(b[0]) == int
    assert type(b[1]) == int

    return a[0] >= b[0] and a[1] <= b[1]

  @staticmethod
  def intervals_overlap(a: list, b: list):
    assert len(a) == 2
    assert len(b) == 2
    assert type(a[0]) == int
    assert type(a[1]) == int
    assert type(b[0]) == int
    assert type(b[1]) == int

    return a[0] <= b[1] and b[0] <= a[1]

  @staticmethod
  def combine_intervals(intervals):
    """Unites intervals within a list that touch or overlap.
    The given list of intervals must be sorted by the respective beginnings of the intervals.

    Example:
      [[1,5], [5,9], [10, 12], [14, 15]] -> [[1,12], [14,15]]

    Args:
      intervals (list): List of intervals(tuples with begin and end)

    Returns:
      list: Intervals covering the same values as the input, but not touching or overlapping.
    """
    assert all(intervals[i][0] <= intervals[i + 1][0] for i in range(len(intervals) - 1)
               ), "Interval array must be sorted"  # checks if intervals are sorted

    result = []
    for begin, end in sorted(intervals):
      if result and result[-1][1] >= begin - 1:
        result[-1][1] = max(result[-1][1], end)
      else:
        result.append([begin, end])
    return result

  @staticmethod
  def intervalsToSet(intervals):
    return set().union(*[set(range(v[0], v[1] + 1)) for v in intervals if len(v) > 1])

  @staticmethod
  def intervals_extract(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
      group = list(group)
      yield [group[0][1], group[-1][1]]

  @staticmethod
  def setToIntervals(idx_set):
    return list(OffsetHelper.intervals_extract(idx_set))
