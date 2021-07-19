import re
pattern='([0-9,D,T,A][a,m,h,s,c]){1}'
musiques=['1a0a(14)3aDa1a1aDa(13)1a','Da6a4a8a0a7a9aDa1a2a','6h9h4h(15)4s1hAs2s1h']

prog = re.compile(pattern,flags=re.IGNORECASE)
result = prog.findall(musiques[2])

for group in result:
    print(group)
print('head')
