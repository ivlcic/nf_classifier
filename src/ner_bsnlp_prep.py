#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import os
import nf
import nf.args
import nf.data
import argparse

from ner_prep import prep_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NER Data preparation and normalization for Slovene, Croatian and Serbian language')
    parser.add_argument('lang', help='language of the text',
                        choices=['bg', 'cs', 'hr', 'pl', 'ru', 'sl', 'sk', 'uk'], default="sl")
    parser.add_argument(
        '-d', '--data_dir', help='Data output directory', type=nf.args.dir_path,
        default=os.path.join(nf.default_data_dir, 'ner_bsnlp'))
    parser.add_argument(
        '-c', '--corpora_dir', help='Corpora input directory', type=nf.args.dir_path,
        default=nf.default_corpora_dir)
    parser.add_argument(
        '-s', '--data_split',
        help='Data split in % separated with colon: '
             'For example "80:10" would produce 80% train, 10% evaluation and 10% test data set size. ',
        type=str, default='80:10'
    )
    parser.add_argument(
        '-r', '--non_reproducible_shuffle', help='Non reproducible data shuffle.', action='store_true', default=False
    )
    args = parser.parse_args()
    tokenizer = nf.data.get_classla_tokenizer(args.lang) \
        if args.lang in ['bg', 'hr', 'sl', 'sr', 'mk'] \
        else nf.data.get_stanza_tokenizer(args.lang)
    confs = [
        {
            'type': 'bsnlp',
            'zip': 'bsnlp-2017-21.zip',
            'proc_file': 'bsnlp',
            'result_name': args.lang + '_bsnlp',
            'map_filter': {
                'max_seq_len': 128,
                'lang': args.lang,
                'tokenizer': tokenizer
            }
        }
    ]
    prep_data(args, confs)
    nf.data.split_data(args, confs)
