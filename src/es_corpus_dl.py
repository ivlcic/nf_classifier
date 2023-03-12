#!/usr/bin/env python
import argparse
import json
import logging
import os

import nf
import nf.args
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from datetime import datetime
from string import Template


logger = logging.getLogger('corpus_dl')
logger.addFilter(nf.fmt_filter)

source_uuid_map = {
    '1b64e062-3e83-4591-af86-a6e244c45ed5': '10',
    '93045532-f197-4da9-9de0-be6795998d7e': '20',
    'bc20546f-3a11-4061-90c2-2769468cd542': '30',
    '07785797-5184-4963-9813-b6611846740a': '40',
    'a6d81cfb-912a-426c-9bc1-845b52a46fe2': '50',
    '4efb7ff6-78e6-4408-879e-59b9f092a8c9': '60',
    '0360424e-26ac-4ef6-991b-d9756663b44f': '70',
    '279ac7fb-4f94-4a04-8dd3-1546d47eedf7': '80',
    '5a32cddf-b3bd-4698-88c1-1ef661fb43a6': '90',
    'b455f571-f0eb-4691-ad54-7d3dfb88f2bd': '100',
    '76320d8f-543a-4b50-92c0-01453c885fd8': '110',
    'bf1d559b-8c27-4f32-8c2f-991d649527b2': '120',
    '7ff71957-487f-456a-b0bb-db6029b8ba04': '130',
    '6a996b7c-b8ea-4aa9-ab67-d8d558d1cba4': '140',
    '685def26-7b7b-4089-b83c-990c06abd752': '150',
    '9aedb9fd-2914-4d8e-9d63-1dbedd03cb65': '160',
    '396b201f-2a65-44c8-857e-16c2bf79e55f': '170',
    '1b498bcf-87b7-4888-b526-4807088c7738': '180',
    '57537984-e949-45a1-b764-6fe4654aed54': '190',
    '2b7bbd45-0ba5-4632-9716-43515f20bd6a': '200',
    'a2d9c362-e25d-483a-a8f3-3b8d90fe05b9': '210',
    'c94af0ee-1f52-4f45-9eac-34ab40835a2f': '220',
    '17d4af83-486f-4add-bd1e-b719ff6b9c2e': '230',
    '29213ab4-199c-4b11-aa64-eaa37092adf6': '240',
    'd53a5e20-a6dd-4ca5-b989-2b662b028f7b': '250',
    'e1ab11df-a6ad-4628-98e0-439376da009b': '260',
    '2c055505-375b-47a3-a5f0-75c57d3cf9e2': '270',
    'ecd1daa4-1f1b-4259-9b03-105f0ba1ba00': '280',
    '754da261-9aee-4a1a-b9d8-734cd409fabf': '290'
}

__query = '''

'''


if __name__ == "__main__":
    '''
    source venv/bin/activate
    src/es_corpus_dl.py data/nf/si_query.json data/nf/middle_east_si.csv 2015-08-01 2016-04-01
    src/es_corpus_dl.py data/nf/si_query.json data/nf/ukraine_si.csv 2022-02-01 2023-03-12
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Query file path to dump csv data")
    parser.add_argument("file", help="File path to dump csv data")
    parser.add_argument("date_start", help="start date for dump in YYYY-MM-DD format")
    parser.add_argument("date_end", help="end date for dump in YYYY-MM-DD format")

    args = parser.parse_args()
    start_date = datetime.fromisoformat(args.date_start)
    end_date = datetime.fromisoformat(args.date_end)
    es = Elasticsearch(hosts='http://localhost:9200/')

    with open(args.query, 'r', encoding='utf-8') as file:
        req = file.read()
    temp_obj = Template(req)
    df = pd.DataFrame()
    for start, end in nf.DateIter(start_date, end_date, step_sec=3600 * 24 * 7):
        req_str = temp_obj.substitute(date_start=start.isoformat(), date_end=end.isoformat())
        req = json.loads(req_str)
        resp = es.search(index="article", **req)
        for hit in resp['hits']['hits']:
            tmp_df = pd.json_normalize(hit['_source'])
            if 'translations.sl.body' not in tmp_df:
                continue
            if 'translations.sl.title' not in tmp_df:
                continue
            if 'mediaReach' not in tmp_df and 'media.mediaReach' in tmp_df:
                tmp_df['mediaReach'] = tmp_df['media.mediaReach']

            created = datetime.strptime(tmp_df.at[0, 'created'], "%Y-%m-%dT%H:%M:%S.%fZ")
            tmp_df.at[0, 'uuid'] = created.strftime("%Y%m%d%H%M%S")
            tmp_df.at[0, 'media.uuid'] = source_uuid_map[tmp_df.at[0, 'media.uuid']]

            tmp_df['translations.sl.title'] = tmp_df['translations.sl.title'].str.replace(' ', ' ')
            tmp_df['translations.sl.title'] = tmp_df['translations.sl.title'].str.strip()
            tmp_df['translations.sl.body'] = tmp_df['translations.sl.body'].str.strip()

            if 'media.mediaReach' in tmp_df:
                tmp_df = tmp_df.drop(['media.mediaReach'], axis=1)

            tmp_df = tmp_df.drop(['created'], axis=1)
            df = pd.concat([df, tmp_df], ignore_index=True)

        logger.debug("Finished [%s]::[%s] hits:[%s] df.shape:[%s]...", start.isoformat(), end.isoformat(),
                     resp['hits']['total']['value'], df.shape)

    logger.debug("Finished dumping df.shape:[%s] to cvs [%s]...", df.shape, args.file)
    df = df.rename(
        columns={
            "uuid": "id",
            "media.uuid": "source_id",
            "media.name": "source_name",
            "mediaReach": "media_reach",
            "translations.sl.body": "body",
            "translations.sl.title": "title"
        }
    )
    df = df[['source_id', 'source_name', 'id', 'published', 'media_reach', 'title', 'url', 'body']]
    df.reindex()
    df.to_csv(args.file)
