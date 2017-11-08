
1.文本Normalization(归一化、标准化)
  在进行normlization之前首先统计各种字符出现的次数，来确定正则化需要过滤的字符

2.文本tokenizer
所谓tokenize就是把一整句话的字符串分割成词的组合，例如:
"I am a student." --> ["I", "am", "a", "student", "."]

常见的tokenize的难题：
  a.缩写，例如："I'm a student."
  b.数字，例如：123,456.78 90.7% 3/8 11/20/2000
  c.缩略，例如：U.S. i.e.
  d.包含非字母字符，例如：AT&T Micro$oft
  e.带杠的词串，例如：three-year-old so-call
  f.其他，例如：网址、公式等

解决方法：
  a.识别分数的正则表达式：[0-9]+ / [0-9]+
  b.识别百分数的正则表达式：([+ | -]) ? [0-9]+ ( . [0-9]* ) ? %
  c.识别十进制数字的正则表达式：( [0-9]+( , )? )+ ( . [0-9]+ )?

\*\*看一下keras的tokenizer

3.词法分析
  a.tokenization：见文本tokenizer
  b.Lemmatization(词形还原)：takes -> take + ~s, took -> take + ~ed
  c.Stemming(词干)：tokenization -> token + ~ize + ~ion
