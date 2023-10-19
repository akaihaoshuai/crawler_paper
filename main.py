import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request
from tqdm import tqdm
import re
import json
from utils import scrape_table,get_arxiv_id_v2,get_abstract_from_arxiv_url,download_arxiv_paper,sort_dict,save_dict2json

def translate_en2zh(model, tokenizer, abstract):
    if model and tokenizer:
        try:
            input_ids = tokenizer.encode(abstract, return_tensors="pt")
            outputs = model.generate(input_ids=input_ids, max_length=1024)
            chinese_abstract = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f'中文：{chinese_abstract}')
            return chinese_abstract
        except:
            print("An exception occurred")
    return None


def process_table(dataframes,args,model,tokenizer,download_dir):
    paper_num = 0
    class_type_dict = dict()
    from collections import Counter
    title_key_word_c = Counter()
    # 打印每个表格的数据,从ArXiv中获取论文ID
    for i, df in enumerate(dataframes):
        paper_num += df.shape[0] - 1
        for idx in range(1, df.shape[0], 1):
            print(f'Table: ({i}/{len(dataframes)}). row: ({idx}/{df.shape[0]}) \n')

            class_type = df[0][idx].replace("：", ": ").replace("/", ",")
            paper_title = df[1][idx].replace("：", ": ")

            from string import digits
            class_type = class_type.translate(str.maketrans('', '', digits))  # remove digits
            if class_type in class_type_dict:
                class_type_dict[class_type] += 1
            else:
                class_type_dict[class_type] = 1

            title_key_word_c.update(re.split(':|：| |,', paper_title))

            print(f'class_type: {class_type}. \npaper name: {paper_title}')
            arxiv_file = open(f'{download_dir}{class_type.replace(": ", "_")}.txt', 'a', encoding='utf-8')

            # 构建ArXiv搜索URL
            arxiv_id, abstract = get_arxiv_id_v2(paper_title)
            arxiv_file.write(f'\ntitle: {paper_title}\n')
            chinese_title = translate_en2zh(model, tokenizer, paper_title)
            if chinese_title is not None:
                arxiv_file.write(f'中文名：{chinese_title}\n')

            abs_url = f'http://arxiv.org/abs/{arxiv_id}'
            pdf_url = f'http://arxiv.org/pdf/{arxiv_id}.pdf'
            print(f'link：{abs_url}')
            print(f'link：{pdf_url}')
            arxiv_file.write(f'link：{abs_url}\n')
            arxiv_file.write(f'link：{pdf_url}\n')

            if arxiv_id is None:
                find_err_file = open(f'{download_dir}find_err.txt', 'a', encoding='utf-8')
                find_err_file.write(f'paper_title：{paper_title}\n')
                find_err_file.close()

            if abstract is None:
                # 发送GET请求获取页面内容
                abstract = get_abstract_from_arxiv_url(abs_url)

            if args.download_paper is True and arxiv_id is not None:
                if args.download_paper_list is None:  # If None, download all
                    download_arxiv_paper(pdf_url, download_dir, class_type, paper_title)
                else:
                    for download_ in args.download_paper_list:
                        lower_name = paper_title.replace('：', '_').replace(': ', '_').lower()
                        lower_type = class_type.replace('：', '_').replace(': ', '_').lower()
                        if download_ in lower_type or download_ in lower_name:
                            download_arxiv_paper(pdf_url, download_dir, class_type, paper_title)
                            break

            if abstract:
                abstract = abstract.split('\n')[0]
                print(f'abstract：{abstract}')
                arxiv_file.write(f'abstract：{abstract}\n')

                # 翻译成中文
                chinese_abstract = translate_en2zh(model, tokenizer, abstract)
                if chinese_abstract is not None:
                    arxiv_file.write(f'摘要：{chinese_abstract}\n')

            arxiv_file.close()
            print("\n")

    save_dict2json(sort_dict(class_type_dict),f'{download_dir}class_type.json')
    save_dict2json(sort_dict(dict(title_key_word_c)),f'{download_dir}paper_title_vocabulary.json')

    return paper_num

def download_iccv2023(args,model,tokenizer):
    download_dir = './ICCV2023/'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # 使用你的URL替换下面的URL
    url = "https://iccv2023.thecvf.com/main.conference.program-107.php"
    # 从网页中抓取表格
    dataframes = scrape_table(url)
    # 处理表格
    paper_num = process_table(dataframes,args,model,tokenizer,download_dir)

    print(f'iccv2023 paper total num: {paper_num}')
    return paper_num

def download_from_openaccess_url(url,args,model,tokenizer,download_dir):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('dt', class_='ptitle')

    openaccess_url = "https://openaccess.thecvf.com/"
    for idx, result in enumerate(results):  # 对于每一个论文标题
        print(f'\npaper idx：({idx}/{len(results)})')
        temp = result.find_all('a')[0]
        paper_title = temp.text.strip().replace('：', '_')
        paper_link = openaccess_url + temp['href']

        arxiv_file = open(f'{download_dir}paper_list_{idx//500}.txt', 'a', encoding='utf-8')
        print(f'title：{paper_title}')
        arxiv_file.write(f'\ntitle: {paper_title}\n')
        chinese_title = translate_en2zh(model, tokenizer, paper_title)
        if chinese_title is not None:
            arxiv_file.write(f'中文名：{chinese_title}\n')

        paper_response = requests.get(paper_link)
        paper_soup = BeautifulSoup(paper_response.text, 'html.parser')
        paper_content = paper_soup.find_all('div', id='content')[0]

        print(f'link：{paper_link}')
        arxiv_file.write(f'link：{paper_link}\n')

        paper_links = paper_content.find_all('dd')[1].find_all('a')
        pdf_url=None
        supplemental_url = None
        arxiv_url = None
        for link in paper_links:
            if link.text.strip() == 'pdf':
                pdf_url = openaccess_url + link['href']
            if link.text.strip() == 'supp':
                supplemental_url = openaccess_url + link['href']
            if link.text.strip() == 'arXiv':
                arxiv_url = link['href']

        if pdf_url is not None:
            arxiv_file.write(f'openaccess pdf：{pdf_url}\n')
        else:
            if arxiv_url is not None:
                pdf_url = arxiv_url.replace('abs','pdf')+'.pdf'
                arxiv_file.write(f'arxiv pdf：{pdf_url}\n')
        # if supplemental_url is not None:
        #     arxiv_file.write(f'supplemental：{supplemental_url}\n')
        # if arxiv_url is not None:
        #     arxiv_file.write(f'arxiv：{arxiv_url}\n')

        paper_authors = paper_content.find_all('div', id='authors')[0].text.strip()
        print(f'author：{paper_authors}')
        arxiv_file.write(f'author：{paper_authors}\n')

        if args.download_paper is True:
            if args.download_paper_list is None:  # If None, download all
                download_arxiv_paper(pdf_url, download_dir, None, paper_title)
            else:
                for download_ in args.download_paper_list:
                    lower_name = paper_title.replace('：', '_').replace(': ', '_').lower()
                    if download_ in lower_name:
                        download_arxiv_paper(pdf_url, download_dir, None, paper_title)
                        break


        paper_abstract = paper_content.find_all('div', id='abstract')[0].text.strip()
        print(f'abstract：{paper_abstract}')
        arxiv_file.write(f'abstract：{paper_abstract}\n')
        chinese_abstract = translate_en2zh(model, tokenizer, paper_abstract)
        if chinese_abstract is not None:
            arxiv_file.write(f'摘要：{chinese_abstract}\n')

        arxiv_file.close()

    return len(results)


def download_iccv2021(args,model,tokenizer):
    download_dir = './ICCV2021/'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    url = "https://openaccess.thecvf.com/ICCV2021?day=all"
    paper_num = download_from_openaccess_url(url,args,model,tokenizer,download_dir)
    print(f'iccv2021 paper total num: {paper_num}')

def download_cvpr2023(args,model,tokenizer):
    download_dir = './ICVPR2023/'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    url = "https://openaccess.thecvf.com/CVPR2023?day=all"
    paper_num = download_from_openaccess_url(url,args,model,tokenizer,download_dir)
    print(f'cvpr2023 paper total num: {paper_num}')

def download_cvpr2022(args,model,tokenizer):
    download_dir = './ICVPR2022/'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    url = "https://openaccess.thecvf.com/CVPR2022?day=all"
    paper_num = download_from_openaccess_url(url,args,model,tokenizer,download_dir)
    print(f'cvpr2022 paper total num: {paper_num}')

def download_cvpr2021(args,model,tokenizer):
    download_dir = './ICVPR2021/'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    url = "https://openaccess.thecvf.com/CVPR2021?day=all"
    paper_num = download_from_openaccess_url(url,args,model,tokenizer,download_dir)
    print(f'cvpr2021 paper total num: {paper_num}')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--download_paper", type=bool, default=True, help="download paper")
    parser.add_argument("--download_paper_list", type=list, default=['single image', 'single-image', 'novel view', 'novel-view'], help="Download the specified paper, which can refer to the specified category or specified content. If None, download all")
    parser.add_argument("--trans_abstract", type=bool, default=True, help="translate paper")
    args = parser.parse_args()

    model_path = './opus-mt-en-zh/'
    if args.trans_abstract and os.path.exists(model_path):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        model = None
        tokenizer = None

    paper_num = download_iccv2023(args,model,tokenizer)
    paper_num += download_iccv2021(args,model,tokenizer)
    paper_num += download_cvpr2023(args,model,tokenizer)
    paper_num += download_cvpr2022(args,model,tokenizer)
    paper_num += download_cvpr2021(args,model,tokenizer)
    print(f'paper total num: {paper_num}')

