import en_core_web_trf

nlp = en_core_web_trf.load()
nlp.add_pipe("merge_noun_chunks")
nlp.add_pipe("merge_entities")

class ConjTermSupplementer():
  @staticmethod
  def merge_noun_chunks(token):
    result = []
    if token.dep_ in ['predet', 'det', 'nummod', 'npadvmod', 'advmod', 'compound', 'case']:
      result.append(token)
    elif token.dep_ == 'amod':
      amod_tokens = [token]
      while len(amod_tokens) > 0:
        amod_token = amod_tokens.pop()
        result.append(amod_token)
        for child in amod_token.children:
          if child.dep_ == 'amod':
            amod_tokens.append(child)
    elif token.dep_ == 'poss':
      result.append(token)
      for cur_child in token.children:
        if cur_child.dep_ == 'case':
          result.append(cur_child)
    return result

  @staticmethod
  def automat1(token, debug=False):
    """[pobj, dobj, nsubj, nsubjpass, appos, attr, conj]"""
    state = 0
    result = []
    success = False
    loop = True

    while loop:
      if len(list(token.children)) == 0:
        loop = False
        success = True

      if state == 0:
        if debug:
          print('State:', state)

        found = False

        for child in token.children:
          result += ConjTermSupplementer.merge_noun_chunks(child)

          if child.dep_ == 'prep':
            result.append(child)
            token = child
            state = 1
            found = True

        if not found:
          loop = False
          success = True
          break

      if state == 1:
        if debug:
          print('State:', state)

        found = False
        for child in token.children:

          if child.dep_ in ['pobj', 'dobj']:
            result.append(child)
            token = child
            state = 2
            found = True
            continue

          if child.dep_ == 'pcomp':
            result.append(child)
            token = child
            state = 3
            found = True
            continue

        if not found:
          loop = False
          success = False
          break

      if state == 2:
        if debug:
          print('State:', state)

        found = False
        for child in token.children:
          result += ConjTermSupplementer.merge_noun_chunks(child)

          if child.dep_ == 'prep':
            result.append(child)
            token = child
            state = 1
            found = True

        if not found:
          success = True
          loop = False
          break

      if state == 3:
        if debug:
          print('State:', state)

        found = False
        for child in token.children:
          result += ConjTermSupplementer.merge_noun_chunks(child)

          if child.dep_ == 'prep':
            result.append(child)
            token = child
            state = 1
            found = True
            continue

          if child.dep_ in ['pobj', 'dobj']:
            result.append(child)
            token = child
            state = 2
            found = True
            continue

        if not found:
          loop = False
          success = True
          break

    return result

  @staticmethod
  def automat2(token, debug=False):
    """[dobj, pobj]"""
    state = 0
    result = []
    success = False
    loop = True

    while loop:
      if len(token.head) == 0:
        loop = False
        success = True

      if state == 0:
        if debug:
          print('State:', state)

        found = False

        for child in token.children:
          result += ConjTermSupplementer.merge_noun_chunks(child)

        if token.head.dep_ == 'pcomp':
          result.insert(0, token.head)
          token = token.head
          found = True
          continue

        if not found:
          loop = False
          success = True
          break

    return result

  @staticmethod
  def automat3(token, debug=False):
    """[prep, conj]"""
    state = 0
    result = []
    success = False
    loop = True

    while loop:
      if len(list(token.children)) == 0:
        loop = False
        success = True

      if state == 0:
        if debug:
          print('State:', state)

        found = False
        for child in token.children:

          if child.dep_ in ['pobj', 'dobj']:
            result.append(child)
            token = child
            state = 1
            found = True
            continue

          if child.dep_ == 'pcomp':
            result.append(child)
            token = child
            state = 2
            found = True
            continue

        if not found:
          loop = False
          success = False
          break

      if state == 1:
        if debug:
          print('State:', state)

        found = False
        for child in token.children:
          result += ConjTermSupplementer.merge_noun_chunks(child)

          if child.dep_ == 'prep':
            result.append(child)
            token = child
            state = 0
            found = True

        if not found:
          success = True
          loop = False
          break

      if state == 2:
        if debug:
          print('State:', state)

        found = False
        for child in token.children:
          result += ConjTermSupplementer.merge_noun_chunks(child)

          if child.dep_ == 'prep':
            result.append(child)
            token = child
            state = 0
            found = True
            continue

          if child.dep_ in ['pobj', 'dobj']:
            result.append(child)
            token = child
            state = 1
            found = True
            continue

        if not found:
          loop = False
          success = True
          break

    return result

  @staticmethod
  def automat4(token, debug=False):
    """[acomp, conj]"""
    state = 0
    result = []
    success = False
    loop = True

    while loop:
      if len(list(token.children)) == 0:
        loop = False
        success = True

      if state == 0:
        if debug:
          print('State:', state)

        found = False

        for child in token.children:
          if child.dep_ == 'xcomp':
            result.append(child)
            token = child
            state = 1
            found = True
            break

        if not found:
          loop = False
          success = True
          break

      if state == 1:
        if debug:
          print('State:', state)

        found = False
        for child in token.children:
          if child.dep_ == 'aux':
            result.insert(0, child)
            token = child
            found = True
            break

        if not found:
          loop = False
          success = True
          break

    return result


  @staticmethod
  def get_conj_term_supplements(conj_term, debug=False):
    token = conj_term
    result = []

    if token.dep_ in ['pobj', 'dobj', 'nsubj', 'nsubjpass', 'appos', 'oprd', 'npadvmod', 'attr','conj', 'advcl']:
      result += ConjTermSupplementer.automat1(token, debug=debug)

    if token.dep_ in ['pobj', 'dobj']:
      result += ConjTermSupplementer.automat2(token, debug=debug)

    if token.dep_ in ['prep', 'conj']:
      result += ConjTermSupplementer.automat3(token, debug=debug)

    if token.dep_ in ['acomp', 'conj']:
      result += ConjTermSupplementer.automat4(token, debug=debug)

    return list(set(result))
