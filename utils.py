import pandas as pd

def uniprot2gene(uniprotIDs):

    import urllib, urllib2

    url = 'https://www.uniprot.org/uploadlists/'

    params = {
        'from': 'ID',
        'to': 'GENENAME',
        'format': 'tab',
        'query': '\t'.join(uniprotIDs)
    }

    data = urllib.urlencode(params)
    request = urllib2.Request(url, data)
    contact = ""
    request.add_header('User-Agent', 'Python %s' % contact)
    response = urllib2.urlopen(request)
    page = response.read(200000)
    result_df = parse_page(page, '\n', '\t')
#    return str(",".join(list(result_df['To'])))
    return result_df[['From', 'To']]

def parse_page(page, row_sep, delimiter):

    for i, row in enumerate(page.split(row_sep)):
        if i == 0:
            df = pd.DataFrame(columns=row.split(delimiter))
        elif not len(row):
            continue
        else:
            df.loc[i-1] = row.split(delimiter)

    return df

