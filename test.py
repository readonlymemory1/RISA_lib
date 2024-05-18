from Sam_train import train_tool

# 시계열 데이터 생성 (간단한 예제)
sentence_data = '''
Do not go gentle into that good night,
Old age should burn and rave at close of day;
Rage, rage against the dying of the light.

Though wise men at their end know dark is right,
Because their words had forked no lightning they
Do not go gentle into that good night.

Good men, the last wave by, crying how bright
Their frail deeds might have danced in a green bay,
Rage, rage against the dying of the light.

Wild men who caught and sang the sun in flight,
And learn, too late, they grieved it on its way,
Do not go gentle into that good night.

Grave men, near death, who see with blinding sight
Blind eyes could blaze like meteors and be gay,   
Rage, rage against the dying of the light.

And you, my father, there on the sad height,
Curse, bless, me now with your fierce tears, I pray.
Do not go gentle into that good night.
Rage, rage against the dying of the light.
'''

cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', sentence_data)

tools = train_tool(cleaned_text)
index2word = tools.i2w()
word2index = tools.w2i()

incoder = lambda x:word2index[x]

for data