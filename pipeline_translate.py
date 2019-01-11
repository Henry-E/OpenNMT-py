#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse
import re
import copy

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts


def main(opt):
    # regexes
    match_split_chars = re.compile(r"(“|”|newwline|staart\w*|ennd\w*)")

    content_selection_opt = copy.deepcopy(opt)
    # TODO not sure whether to store model as a string or a list
    content_selection_opt.model = [opt.models[0]]
    content_selection_opt.models = [opt.models[0]]
    content_selection_translator = build_translator(content_selection_opt,
                                                    report_score=False)
    surface_realization_opt = copy.deepcopy(opt)
    surface_realization_opt.model = [opt.models[1]]
    surface_realization_opt.models = [opt.models[1]]
    surface_realization_translator = build_translator(surface_realization_opt,
                                                      report_score=False)
    # TODO add an opt for generating num stories
    for _ in range(2):
        story_so_far = ['starrtWP']
        num_lines = 0
        while num_lines < 200:
            num_lines += 1
            _, deep_next_line_batch = content_selection_translator.translate(
                src=[' '.join(story_so_far)],
                batch_size=1)
            deep_next_line = deep_next_line_batch[0][0]
            if match_split_chars.match(deep_next_line):
                story_so_far.append(deep_next_line)
                continue
            if deep_next_line == 'end_of_story':
                story_so_far.append(deep_next_line)
                break
            deep_story_input = story_so_far + ['start_deep_ud'] + \
                               [deep_next_line]
            _, story_next_line = surface_realization_translator.translate(
                src=[' '.join(deep_story_input)],
                batch_size=1)
            story_so_far.append(story_next_line[0][0])
        print(num_lines, ' '.join(story_so_far))

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='translate.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
