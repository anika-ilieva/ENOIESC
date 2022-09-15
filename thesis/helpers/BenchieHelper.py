import pickle
import pandas as pd

class BenchieHelper:
  @staticmethod
  def fix_format(extraction_path, dataset_path):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    path = extraction_path + '/extraction.txt'
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ["sentence", "score", "predicate", "arg1", 'arg2', 'arg3', 'arg4']

    with open(extraction_path + '/extraction_benchie_format.txt', 'w') as out_file:
      i = 0

      for number in range(1, len(data) + 1):
        while i < len(df) and df.iloc[i, 0] == data[number-1]:
          pred = str(df.iloc[i, 2])
          arg1 = str(df.iloc[i, 3])
          arg2 = str(df.iloc[i, 4])
          arg3 = str(df.iloc[i, 5])
          arg4 = str(df.iloc[i, 6])

          if arg1 == 'nan':
              arg1 = ''
          if arg2 == 'nan':
              arg2 = ''
          if arg3 == 'nan':
              arg3 = ''
          if arg4 == 'nan':
              arg4 = ''
          if pred == 'nan':
              pred = ''

          args = arg2 + arg3 + arg4

          line_output = [str(number), arg1, pred, args]

          out_file.write("\t".join(line_output) + '\n')
          i += 1
