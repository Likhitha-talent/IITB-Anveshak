import re
import requests

def test():
    s = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    r = s.get('https://docs.google.com/uc?export=download&id=1A24PTWWji509iDwf50EkJuTcwyb13RDH', headers=headers)
    m = re.search(r'name="uuid" value="([^"]+)"', r.text)
    if m:
        uuid = m.group(1)
        url2 = "https://drive.usercontent.google.com/download"
        params = {'id': '1A24PTWWji509iDwf50EkJuTcwyb13RDH', 'export': 'download', 'confirm': 't', 'uuid': uuid}
        r2 = s.get(url2, params=params, headers=headers)
        print("Response text:", r2.text)

if __name__ == '__main__':
    test()
