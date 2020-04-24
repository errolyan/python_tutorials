from io import BytesIo

f = BytesIO()
print(f.write('中国'.encode('utf-8')))
print(f.getvalue())