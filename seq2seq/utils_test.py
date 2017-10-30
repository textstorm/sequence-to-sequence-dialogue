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