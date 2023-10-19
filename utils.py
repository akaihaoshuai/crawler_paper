
import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request
import os


def scrape_table(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.find_all('table')  # 找到所有的表格

    dataframes = []  # 用于存储所有表格的数据
    for table in tables[3:]:  # 对于每一个表格
        data = []
        for row in table.find_all('tr')[1:]:  # 对于表格中的每一行
            cols = row.find_all('td')  # 找到所有的列
            # cols = [ele.text.strip()+'\n' for ele in cols]  # 获取每一列中的全部数据
            cols = [cols[1].text.strip()]+[cols[2].next]  # 获取列中的数据
            data.append([ele for ele in cols if ele])  # 添加到数据列表中

        df = pd.DataFrame(data)  # 创建一个pandas DataFrame
        dataframes.append(df)  # 添加到数据框列表中

    return dataframes

def get_arxiv_id_v0(paper_title):
    query = urllib.parse.quote(paper_title)
    url = f"https://arxiv.org/search/?query={paper_title}&searchtype=title"

    # 获取搜索结果并解析ArXiv ID
    with urllib.request.urlopen(url) as response:
        xml_data = response.read()
        id_start = xml_data.find(b'http://arxiv.org/abs/') + 19
        id_end = xml_data.find(b'v', id_start) - 1
        arxiv_id = xml_data[id_start:id_end].decode('utf-8').replace('.', '/')
    return arxiv_id

def get_arxiv_id_v1(paper_title):
    base_url = "http://export.arxiv.org/api/query?"

    # 构建查询参数
    query = f"ti:{paper_title}"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 5
    }

    response = requests.get(base_url, params=params)
    # 解析响应结果
    if response.status_code == 200:
        xml_data = response.text
        start_index = xml_data.find("<id>") + 4
        end_index = xml_data.find("</id>")
        arxiv_id = xml_data[start_index:end_index]
        return arxiv_id
    else:
        return None

def get_arxiv_id_v2(paper_title):
    search_url = f"https://arxiv.org/search/?query={paper_title}&searchtype=title"

    # 发送HTTP请求并获取响应
    response = requests.get(search_url)

    # 解析HTML响应
    soup = BeautifulSoup(response.text, 'html.parser')

    # 提取搜索结果中的论文标题和arXiv ID
    results = soup.find_all('li', class_='arxiv-result')
    for result in results:
        title = result.find('p', class_='title').text.strip()

        import re
        arxiv_id_str = result.find('p', class_='list-title').text.strip()
        s1 = re.compile(r'[arXiv:](.*?)[\n]', re.S)
        arxiv_id=re.findall(s1, arxiv_id_str)[0][5:]
        abstract = result.find('span', class_='abstract-full').text.strip()

        return arxiv_id, abstract

    return None,None

def sort_dict(dict_in):
    sorted_dict = sorted(dict_in.items(), key=lambda x: x[1], reverse=True)
    dict_out = dict()
    for type in sorted_dict:
        dict_out[type[0]] = type[1]

    return dict_out

def save_dict2json(dict_in, data_str):
    import json
    jsonstr = json.dumps(dict_in,indent=2)
    filename = open(data_str, 'w')  # dict转josn
    filename.write(jsonstr)
    filename.close()


def get_abstract_from_arxiv_url(abs_url):
    response = requests.get(abs_url)
    if response.status_code == 200:
        # 使用BeautifulSoup解析页面内容
        soup = BeautifulSoup(response.content, 'html.parser')

        # 提取摘要信息
        abstract_element = soup.find('blockquote', class_='abstract')
        if abstract_element:
            abstract = abstract_element.text.strip()
            print(f'摘要：{abstract}')

            return abstract
        else:
            print(f'无法找到摘要：{abs_url}')
            return None
    else:
        print(f'无法访问论文页面：{abs_url}')
        return None


def download_arxiv_paper(pdf_url,download_dir,class_type,file_name):
    try:
        name = file_name.replace('：', '_').replace(': ', '_')

        if class_type is not None:
            import shutil
            dir_path = f'{download_dir}{class_type}'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            file_path = f'{download_dir}{class_type}/{name}.pdf'
        else:
            file_path = f'{download_dir}{name}.pdf'

        if not os.path.exists(file_path):
            fp = open(f'{file_path}', "wb")

            # r = requests.head(pdf_url)
            # r = requests.get(pdf_url, stream=True, timeout=None)
            # for i in r.iter_content(2048):
            #     fp.write(i)

            u = urllib.request.urlopen(pdf_url)
            while True:
                buffer = u.read(8192)
                if buffer:
                    fp.write(buffer)
                else:
                    break

            fp.close()
            print(f'download <<{file_name}>> ok.')
    except:
        print(f'download <<{file_name}>> error!!!!')
