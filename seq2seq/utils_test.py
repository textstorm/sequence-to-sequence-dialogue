import nltk

#1.tokenizer test
nltk.word_tokenize("I'm a 1,234 student.") 
#['I', "'m", 'a', '1,234', 'student', '.']
nltk.word_tokenize('That U.S.A. poster-print costs $12.40...')
#['That', 'U.S.A.', 'poster-print', 'costs', '$', '12.40', '...']  $12.40 shouldn't be splited
nltk.word_tokenize('I didn\'t have a pencil.')
#['I', 'did', "n't", 'have', 'a', 'pencil', '.']
nltk.word_tokenize('12,34.78 90.34% 12/20/2000 3/8')
#['12,34.78', '90.34', '%', '12/20/2000', '3/8']   90.34% shouldn't be splited
nltk.word_tokenize('U.S. i.e.')
#['U.S.', 'i.e', '.']  if i.e. in the end of a sentence tokenize may be error
nltk.word_tokenize('U.S. i.e. and.')
#['U.S.', 'i.e', '.', 'and', '.']  it's right in general
nltk.word_tokenize('AT&T Micro$oft')
#['AT', '&', 'T', 'Micro', '$', 'oft']
#when the word contains non-alphabetic characters, tokenize error
nltk.word_tokenize('three-year-old so-call')
#['three-year-old', 'so-call']
nltk.word_tokenize('I want to New York')
#['I', 'want', 'to', 'New', 'York']  In general New York can be treat as one word
nltk.word_tokenize("what is  rock 'n' roll?")
#['what', 'is', 'rock', "'n", "'", 'roll', '?']    rock 'n' roll

#2.filter word
#remove all non-alphabetic characters include number and space
#do not advocate use in dialogue data cleaning
re.sub(r'\W+', '', "micro$oft and")       #'microoftand'

#remove all non-alphabetic characters except .!?
#keep space but maybe split the word
re.sub(r"[^a-zA-Z.!?]+", r" ", "micro$oft and")       #'micro oft and'

#remove all non-alphabetic non-number characters except .!?spce
#directly remove the non-character in the word
re.sub(r"[^a-zA-Z0-9.!? ]+", r"", "micro$oft and")       #'microoft and'

#
re.sub(r"[^a-zA-Z0-9\' ]+", r"", sentence)

#maybe faster
pattern = re.compile('([^\s\w\']|_)+')
pattern.sub('', sentence)

#3.character statistics
from collections import Counter
def char_statistics(sentences, counter):
  for sentence in sentences:
    counter.update(sentence)
  return counter

counter = Counter()
char_counter = char_statistics(sentences, counter)