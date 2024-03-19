def convert(text):
    if text.endswith(":)"):
        new_text = text.replace(":)","🙂")
    elif text.endswith(":("):
        new_text = text.replace(":(","🙁")
    return new_text

prompt =input()
print(convert(prompt))
    