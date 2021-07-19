import requests
proxies = {
    'http': 'socks5://localhost:9050',
    'https': 'socks5://localhost:9050'
}
url = 'http://httpbin.org/ip'
print(requests.get(url, proxies=proxies).text)

print(requests.get(url).text)