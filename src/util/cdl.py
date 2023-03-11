#!/usr/bin/env python
import argparse
import logging
import os

import nf
import nf.args


logger = logging.getLogger('train')
logger.addFilter(nf.fmt_filter)

mmap = {
    
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
