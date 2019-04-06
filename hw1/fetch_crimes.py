import requests

url_2017 = 'https://data.cityofchicago.org/api/views/d62x-nvdr/rows.csv?accessType=DOWNLOAD'
url_2018 = 'https://data.cityofchicago.org/api/views/3i3m-jwuy/rows.csv?accessType=DOWNLOAD'

def download_csv(url, filename):
    with open(filename, 'w') as f:
        r = requests.get(url)
        f.write(r.text)

def download_all_csvs():
    download_csv(url_2017, 'data/crimes2017.csv')
    download_csv(url_2018, 'data/crimes2018.csv')

if __name__ == "__main__":
    download_all_csvs()
