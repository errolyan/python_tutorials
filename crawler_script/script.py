#!/usr/bin python
# -*-coding:utf-8-*-
#__Author__ = "ErrolYan"



from urllib import request, parse
# Base URL being accessed
url = 'http://httpbin.org/get'
# Dictionary of query parameters (if any)
parms = {
'name1' : 'value1',
'name2' : 'value2'
}
# Encode the query string
querystring = parse . urlencode(parms)
# Make a GET request and read the response
u = request . urlopen(url + '?' + querystring)
resp = u . read()
print(resp)





import requests
# Base URL being accessed
url = 'http://httpbin.org/post'
# Dictionary of query parameters (if any)
parms = {
'name1' : 'value1',
'name2' : 'value2'
}
# Extra headers
headers = {
'User-agent' : 'none/ofyourbusiness',
'Spam' : 'Eggs'
}
resp = requests . post(url, data = parms, headers = headers)
# Decoded text returned by the request
text = resp . text
print(text)
content = resp.content
print(content)

import requests
# First request
resp1 = requests . get(url)
...
# Second requests with cookies received on first requests
resp2 = requests . get(url, cookies = resp1 . cookies)

print(resp2)

