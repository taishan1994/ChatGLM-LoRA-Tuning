"""
这里可以自定义各种解码策略
"""

def get_entities(tokens):
    res = []
    length = len(tokens)
    start = 0
    end = 0
    while end < length:
        while end < length and tokens[end] != "<n>":
            end += 1
        if start != end:
            tmp = "".join(tokens[start:end])
            tmp = tmp.split("_")
            if tmp[0] != "":
                res.append(tmp)
        start = end+1
        end += 1
    return res